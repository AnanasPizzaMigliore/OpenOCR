import os
import cv2
import json
import numpy as np

def visualize_rotated_text_labels_to_files(label_lines, save_dir="./vis_results"):
    """
    Draw polygon annotations on images and save to disk.

    Args:
        label_lines (list[str]): Lines in the format:
            "<image_path>\t[{'points': [[x1, y1], [x2, y2], ...]}]"
        save_dir (str): Directory to save output images.
    """

    os.makedirs(save_dir, exist_ok=True)

    for line in label_lines:
        line = line.strip()
        if not line:
            continue

        try:
            img_path, ann_str = line.split('\t', 1)
        except ValueError:
            print(f"[WARN] Skipping malformed line: {line}")
            continue

        try:
            anns = json.loads(ann_str)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode error for {img_path}: {e}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        # Draw polygons
        for ann in anns:
            for (x, y) in ann.get("points", []):
                center = (int(x), int(y))
                # Bright red outer circle (ring)
                cv2.circle(img, center, radius=10, color=(0, 0, 255), thickness=3)
                # White inner circle
                cv2.circle(img, center, radius=4, color=(255, 255, 255), thickness=-1)


        # Save result
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)
        print(f"[INFO] Saved visualization: {save_path}")


# Example usage
with open("/scratch/penghao/OpenOCR/det_results/det_results.txt") as f:
    lines = f.readlines()
visualize_rotated_text_labels_to_files(lines, save_dir="/scratch/penghao/OpenOCR/det_results/images")

