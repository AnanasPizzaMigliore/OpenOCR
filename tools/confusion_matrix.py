import json
import os
import numpy as np
import cv2

# ========== Utility Functions ==========

def iou_polygon(poly1, poly2):
    """Compute IoU between two polygons (lists of [x,y])."""
    poly1 = np.array(poly1, dtype=np.int32).reshape(-1, 2)
    poly2 = np.array(poly2, dtype=np.int32).reshape(-1, 2)

    # Skip degenerate polygons
    if poly1.shape[0] < 3 or poly2.shape[0] < 3:
        return 0.0

    h = int(max(poly1[:, 1].max(), poly2[:, 1].max()) + 10)
    w = int(max(poly1[:, 0].max(), poly2[:, 0].max()) + 10)
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask1, [poly1], 1)
    cv2.fillPoly(mask2, [poly2], 1)

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


def filter_small_boxes(polygons, min_area=100):
    """Remove polygons smaller than min_area."""
    filtered = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        area = cv2.contourArea(pts)
        if area >= min_area:
            filtered.append(poly)
    return filtered


def detection_confusion_matrix(preds, gts, iou_thresh=0.5):
    """Compute TP, FP, FN, precision, recall, F1."""
    matched_gt = set()
    TP = 0
    for pred in preds:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou = iou_polygon(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thresh:
            TP += 1
            matched_gt.add(best_idx)
    FP = len(preds) - TP
    FN = len(gts) - TP
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}


def parse_label_lines(lines):
    """Parse PaddleOCR-style label lines."""
    data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            img_path, ann_str = line.split("\t", 1)
            anns = json.loads(ann_str)
            # Force shape consistency
            polygons = []
            for a in anns:
                pts = np.array(a["points"], dtype=np.float32).reshape(-1, 2).tolist()
                polygons.append(pts)
            data[os.path.basename(img_path)] = polygons
        except Exception as e:
            print(f"[WARN] Skipping malformed line: {line[:80]}... ({e})")
    return data


# ========== Main Evaluation ==========

with open("/scratch/penghao/OpenOCR/det_results/det_results.txt") as f:
    lines_det = f.readlines()

with open("/scratch/penghao/datasets/Products-Real-Rotated/evaluation/labels.txt") as f:
    lines_gt = f.readlines()

pred_dict = parse_label_lines(lines_det)
gt_dict = parse_label_lines(lines_gt)

total_TP, total_FP, total_FN = 0, 0, 0

for img_name, gt_polys in gt_dict.items():
    preds = pred_dict.get(img_name, [])
    preds = filter_small_boxes(preds, min_area=2000)  # remove tiny detections

    result = detection_confusion_matrix(preds, gt_polys, iou_thresh=0.5)
    total_TP += result["TP"]
    total_FP += result["FP"]
    total_FN += result["FN"]

precision = total_TP / (total_TP + total_FP + 1e-8)
recall = total_TP / (total_TP + total_FN + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\n=== Global Detection Metrics ===")
print(f"TP = {total_TP}")
print(f"FP = {total_FP}")
print(f"FN = {total_FN}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1-score  = {f1:.4f}")
