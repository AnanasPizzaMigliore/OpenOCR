from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from pathlib import Path
import re
from datetime import datetime
from dateutil import parser as date_parser

# Add project root to path
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse
import numpy as np
import copy
import time
import cv2
import json
from PIL import Image, ImageDraw, ImageFont

# Import internal tools
try:
    from tools.utils.utility import get_image_file_list, check_and_read
    from tools.infer_rec import OpenRecognizer
    from tools.infer_det import OpenDetector
    from tools.engine.config import Config
    from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
    from tools.utils.logging import get_logger
except ImportError:
    print("Error: Could not import internal tools. Make sure you are running this from the project root.")
    sys.exit(1)

logger = get_logger()

# -----------------------------------------------------------------------------
# Date Parser Class
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Date Parser Class (Updated)
# -----------------------------------------------------------------------------
class DateParser:
    def __init__(self):
        # Regex patterns to clean common OCR noise before parsing
        self.noise_patterns = [
            (r'[|\[\]\{\}]', ''),             # Remove brackets/pipes often confused for 1 or l
            (r'[\u4e00-\u9fa5]', ' '),        # Replace Chinese chars with space
            (r'[^a-zA-Z0-9\s\.\-\/:]', ''),   # Remove special chars except likely date separators
            # Fix spacing in YYYY . MM . DD
            (r'(\d{4})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{1,2})', r'\1-\2-\3'), 
            # Fix spacing in DD . MM . YYYY
            (r'(?<!\d)(\d{1,2})\s*[\./-]\s*(\d{1,2})\s*[\./-]\s*(\d{4})', r'\1-\2-\3'),
        ]

    def clean_text(self, text):
        """Pre-process text to remove OCR artifacts."""
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        for pattern, replacement in self.noise_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()

    def parse(self, text, default_format='%Y-%m-%d'):
        """
        Attempts to parse a date from a string.
        Resolves ambiguity (e.g., 01/05/2024) by picking the date closest to TODAY.
        """
        if not text:
            return None

        clean_text = self.clean_text(text)
        
        # Quick check: if the text is too short or has no numbers, skip
        if len(clean_text) < 6 or not any(char.isdigit() for char in clean_text):
            return None

        candidates = []
        now = datetime.now()

        # --- Strategy 1: Dateutil with Fuzzy Matching ---
        # We try both dayfirst=True (EU/International) and dayfirst=False (US)
        # to capture ambiguous dates like 02/03/2025.
        for df_setting in [True, False]:
            try:
                # fuzzy_with_tokens=True allows us to check if the date covers enough of the string
                # but for simplicity, we stick to standard fuzzy parsing here.
                dt = date_parser.parse(clean_text, fuzzy=True, dayfirst=df_setting)
                candidates.append(dt)
            except (ValueError, OverflowError, TypeError):
                continue

        # --- Strategy 2: Explicit Dense Formats (OCR often misses separators) ---
        # Handle formats like "20251201" or "011225" which dateutil might treat as numbers.
        digits_only = re.sub(r'\D', '', clean_text)
        
        dense_formats = []
        if len(digits_only) == 6:
            dense_formats = ['%y%m%d', '%d%m%y', '%m%d%y'] # YYMMDD, DDMMYY, MMDDYY
        elif len(digits_only) == 8:
            dense_formats = ['%Y%m%d', '%d%m%Y', '%m%d%Y'] # YYYYMMDD, DDMMYYYY, MMDDYYYY

        for fmt in dense_formats:
            try:
                dt = datetime.strptime(digits_only, fmt)
                candidates.append(dt)
            except ValueError:
                continue

        if not candidates:
            return None

        # --- Strategy 3: Selection Logic (Closest to Now) ---
        best_date = None
        min_diff = float('inf')
        
        # Use a set to remove duplicates (e.g. 2024-05-05 is same in US/EU)
        unique_candidates = set(candidates)

        for cand in unique_candidates:
            # Filter out unreasonable years (OCR noise often creates year 1900 or 3000)
            # Adjust range as needed for your specific domain (e.g., Expiration dates)
            if not (2000 <= cand.year <= 2035):
                continue

            # Calculate absolute difference from now
            diff = abs((cand - now).total_seconds())
            
            if diff < min_diff:
                min_diff = diff
                best_date = cand

        if best_date:
            return best_date.strftime(default_format)
        
        return None

# -----------------------------------------------------------------------------
# OpenOCR Class (Modified)
# -----------------------------------------------------------------------------
class OpenOCR(object):
    def __init__(self,
                 det_config_path,
                 rec_config_path,
                 det_model_path=None,
                 rec_model_path=None,
                 backend='torch',
                 drop_score=0.5,
                 det_box_type='quad',
                 device='gpu',
                 crop_padding=5):
        
        # 1. Load Detection Config
        self.cfg_det = Config(det_config_path).cfg
        self.cfg_det['Global']['device'] = device
        if det_model_path:
             self.cfg_det['Global']['checkpoints'] = det_model_path
             self.cfg_det['Global']['pretrained_model'] = det_model_path

        # 2. Load Recognition Config
        self.cfg_rec = Config(rec_config_path).cfg
        self.cfg_rec['Global']['device'] = device
        if rec_model_path:
             self.cfg_rec['Global']['checkpoints'] = rec_model_path
             self.cfg_rec['Global']['pretrained_model'] = rec_model_path

        # 3. Initialize Engines
        self.text_detector = OpenDetector(self.cfg_det, backend=backend)
        self.text_recognizer = OpenRecognizer(self.cfg_rec, backend=backend)
        
        # 4. Initialize Date Parser
        self.date_parser = DateParser()
        
        self.det_box_type = det_box_type
        self.drop_score = drop_score
        self.crop_padding = crop_padding

    def sorted_boxes(self, dt_boxes):
        """Sort text boxes in order from top to bottom, left to right"""
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def expand_box(self, box, img_h, img_w):
        """Expand box by padding pixels"""
        pad = self.crop_padding
        if pad == 0:
            return box

        box = np.array(box, dtype=np.float32)
        center = np.mean(box, axis=0)
        new_box = np.zeros_like(box)

        for i in range(4):
            vec = box[i] - center
            length = np.linalg.norm(vec)
            if length < 1e-6:
                new_box[i] = box[i]
                continue
            unit_vec = vec / length
            new_box[i] = box[i] + unit_vec * pad

        new_box[:, 0] = np.clip(new_box[:, 0], 0, img_w - 1)
        new_box[:, 1] = np.clip(new_box[:, 1], 0, img_h - 1)
        return new_box

    def infer_single_image(self, img_numpy, ori_img, crop_infer=False, rec_batch_num=6, return_mask=False, **kwargs):
        start = time.time()
        img_h, img_w = ori_img.shape[:2]
        
        # --- 1. Detection ---
        if crop_infer:
            dt_boxes = self.text_detector.crop_infer(img_numpy=img_numpy)[0]['boxes']
        else:
            det_res = self.text_detector(img_numpy=img_numpy, return_mask=return_mask, **kwargs)[0]
            dt_boxes = det_res['boxes']
            
        det_time_cost = time.time() - start

        if dt_boxes is None or len(dt_boxes) == 0:
            return None, None, None

        img_crop_list = []
        original_dt_boxes = self.sorted_boxes(dt_boxes)

        # --- 2. Prepare Crops ---
        for bno in range(len(original_dt_boxes)):
            tmp_box = np.array(copy.deepcopy(original_dt_boxes[bno])).astype(np.float32)
            expanded_box = self.expand_box(tmp_box, img_h, img_w)

            if self.det_box_type == 'quad':
                img_crop = get_rotate_crop_image(ori_img, expanded_box)
                #crop_name = f"/scratch/penghao/OpenOCR/output_inference/vis/img_crop_2.jpg"
                #cv2.imwrite(crop_name, img_crop)
            else:
                img_crop = get_minarea_rect_crop(ori_img, expanded_box)
            img_crop_list.append(img_crop)

        # --- 3. Run Recognition (Pass 1 - Normal) ---
        start = time.time()
        rec_res = self.text_recognizer(img_numpy_list=img_crop_list, batch_num=rec_batch_num)
        
        # --- 4. Orientation Check (Pass 2 - Flipped) ---
        flip_candidates = []
        for img in img_crop_list:
            rotated_img = cv2.rotate(img, cv2.ROTATE_180)
            flip_candidates.append(rotated_img)
        
        if len(flip_candidates) > 0:
            flipped_rec_res = self.text_recognizer(img_numpy_list=flip_candidates, batch_num=rec_batch_num)
            
            for i in range(len(rec_res)):
                original_score = rec_res[i]['score']
                original_text = rec_res[i]['text']
                
                flipped_score = flipped_rec_res[i]['score']
                flipped_text = flipped_rec_res[i]['text']
                
                if flipped_score > original_score:
                    logger.info(f"Fixed orientation: '{original_text}'({original_score:.3f}) -> '{flipped_text}'({flipped_score:.3f})")
                    rec_res[i] = flipped_rec_res[i]

        rec_time_cost = time.time() - start

        # --- 5. Filter & Return ---
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(original_dt_boxes, rec_res):
            text, score = rec_result['text'], rec_result['score']
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append([text, score])

        avg_rec_time = rec_time_cost / len(original_dt_boxes) if len(original_dt_boxes) > 0 else 0
        
        time_info = {
            'time_cost': det_time_cost + rec_time_cost,
            'detection_time': det_time_cost,
            'recognition_time': rec_time_cost,
            'avg_rec_time_cost': avg_rec_time
        }
        
        if return_mask:
            return filter_boxes, filter_rec_res, time_info, det_res['mask']
        return filter_boxes, filter_rec_res, time_info

    def __call__(self, img_path, save_dir='e2e_results/', is_visualize=False, **kwargs):
        
        image_file_list = get_image_file_list(img_path)
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, image_file in enumerate(image_file_list):
            img, flag_gif, flag_pdf = check_and_read(image_file)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(image_file)
            
            if img is None:
                logger.info(f"error reading image {image_file}")
                continue

            logger.info(f'Processing {idx+1}/{len(image_file_list)}: {image_file} (padding={self.crop_padding}px)')

            ori_img = img.copy()
            dt_boxes, rec_res, time_dict = self.infer_single_image(img, ori_img, **kwargs)

            if dt_boxes is None:
                logger.info('No text detected.')
                continue

            # Save Result
            res_list = []
            for i in range(len(dt_boxes)):
                text = rec_res[i][0]
                score = float(rec_res[i][1])
                
                # --- APPLY DATE PARSER ---
                parsed_date = self.date_parser.parse(text)
                
                res_dict = {
                    'transcription': text,
                    'points': np.array(dt_boxes[i]).tolist(),
                    'score': score
                }
                
                # Only add 'date' field if a valid date was found
                if parsed_date:
                    res_dict['date'] = parsed_date
                    logger.info(f"  Found Date: {text} -> {parsed_date}")

                res_list.append(res_dict)
            
            save_path = os.path.join(save_dir, 'inference_results.txt')
            time_cost = time_dict['time_cost']
            with open(save_path, 'a', encoding='utf-8') as f:
                output_data = {
                "results": res_list,
                "time_cost": time_cost
                }

                f.write(json.dumps(output_data, ensure_ascii=False) + '\n')

            # Visualization
            if is_visualize:
                self.visualize(ori_img, dt_boxes, rec_res, save_dir, image_file, res_list)

    def visualize(self, img_bgr, dt_boxes, rec_res, save_dir, image_file, res_list=None):
        """ Draw boxes. If a date is found, draw the PARSED date in GREEN. Uses UNIFORM font size. """
        import urllib.request
        
        # 1. Setup Font
        font_path = os.path.join(str(Path.home()), '.cache', 'openocr', 'simfang.ttf')
        if not os.path.exists(font_path):
             os.makedirs(os.path.dirname(font_path), exist_ok=True)
             urllib.request.urlretrieve('https://shuiche-shop.oss-cn-chengdu.aliyuncs.com/fonts/simfang.ttf', font_path)
        
        # 2. Create Transparent Overlay
        image_rgba = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
        overlay = Image.new('RGBA', image_rgba.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # --- UNIFORM FONT SIZE CALCULATION ---
        # Calculate one font size for the whole image based on image height
        # e.g., 2.5% of image height, but at least 15 pixels.
        img_h = img_bgr.shape[0]
        fixed_font_size = max(int(img_h * 0.025), 15)

        try:
            font = ImageFont.truetype(font_path, fixed_font_size, encoding="utf-8")
        except:
            font = ImageFont.load_default()
        
        for idx, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            if score < self.drop_score:
                continue
            
            # Check for parsed date
            parsed_date_str = None
            if res_list and idx < len(res_list):
                parsed_date_str = res_list[idx].get('date')

            is_date = parsed_date_str is not None
            
            # --- COLOR LOGIC ---
            outline_color = (0, 255, 0, 255) if is_date else (255, 0, 0, 255)
            text_bg_color = (0, 255, 0, 150) if is_date else (255, 255, 255, 100)
            text_color = (0, 0, 0, 255) if is_date else (0, 0, 255, 180)

            # Display Text
            display_text = f"DATE: {parsed_date_str}" if is_date else text

            # Draw Box
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            points = [tuple(p) for p in box]
            draw_overlay.polygon(points, outline=outline_color, width=3)
            
            # Text Position
            txt_x = box[0][0]
            # Position text slightly above the box
            txt_y = max(0, box[0][1] - fixed_font_size - 4)
            
            # Draw Text Background & Text
            left, top, right, bottom = draw_overlay.textbbox((txt_x, txt_y), display_text, font=font)
            
            # Draw background rectangle slightly larger than text
            draw_overlay.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=text_bg_color)
            draw_overlay.text((txt_x, txt_y), display_text, fill=text_color, font=font)

        # 3. Save
        image_combined = Image.alpha_composite(image_rgba, overlay)
        image_final = image_combined.convert('RGB')

        vis_dir = os.path.join(save_dir, 'test_images2')
        os.makedirs(vis_dir, exist_ok=True)
        save_vis_path = os.path.join(vis_dir, os.path.basename(image_file))
        image_final.save(save_vis_path)
        logger.info(f"Visualization saved to {save_vis_path}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='OpenOCR Custom Inference with Date Parsing')
    parser.add_argument('--img_path', type=str, required=True, help='Image file or directory')
    parser.add_argument('--det_config', type=str, required=True, help='Path to your Det YAML config')
    parser.add_argument('--rec_config', type=str, required=True, help='Path to your Rec YAML config')
    parser.add_argument('--det_model_path', type=str, default=None, help='Path to your Det .pth')
    parser.add_argument('--rec_model_path', type=str, default=None, help='Path to your Rec .pth')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--save_dir', type=str, default='./output_inference')
    parser.add_argument('--vis', action='store_true', help='Visualize results')
    parser.add_argument('--padding', type=int, default=5, help='Pixels to pad crop (default: 5)')
    
    args = parser.parse_args()

    ocr_engine = OpenOCR(
        det_config_path=args.det_config,
        rec_config_path=args.rec_config,
        det_model_path=args.det_model_path,
        rec_model_path=args.rec_model_path,
        device=args.device,
        crop_padding=args.padding 
    )

    ocr_engine(
        img_path=args.img_path,
        save_dir=args.save_dir,
        is_visualize=args.vis
    )

if __name__ == '__main__':
    main()