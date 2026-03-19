import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from ultralytics import YOLO


# ── IoU & CIoU ────────────────────────────────────────────────────────────────

def box_iou(b1, b2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def box_ciou(b1, b2):
    """
    Compute CIoU between two boxes [x1,y1,x2,y2].
    CIoU = IoU - (rho^2 / c^2) - alpha * v
      rho = center distance, c = diagonal of enclosing box
      v   = aspect ratio consistency term
    """
    iou = box_iou(b1, b2)

    # center points
    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # enclosing box diagonal
    enc_x1 = min(b1[0], b2[0])
    enc_y1 = min(b1[1], b2[1])
    enc_x2 = max(b1[2], b2[2])
    enc_y2 = max(b1[3], b2[3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # aspect ratio term
    import math
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    v = (4 / math.pi ** 2) * (math.atan(w2 / (h2 + 1e-7)) - math.atan(w1 / (h1 + 1e-7))) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha * v
    return float(ciou)


# ── Annotation loader ─────────────────────────────────────────────────────────

def load_gt_boxes(xml_path, img_w, img_h):
    """Return list of [x1,y1,x2,y2] ground-truth boxes from VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        boxes.append([
            float(bb.find("xmin").text),
            float(bb.find("ymin").text),
            float(bb.find("xmax").text),
            float(bb.find("ymax").text),
        ])
    return boxes


# ── Matching: best GT for each prediction ────────────────────────────────────

def match_boxes(pred_boxes, gt_boxes):
    """
    For each predicted box find the GT box with highest IoU.
    Returns list of (pred_box, best_gt_box) pairs.
    """
    pairs = []
    for pred in pred_boxes:
        if not gt_boxes:
            pairs.append((pred, None))
            continue
        best_gt = max(gt_boxes, key=lambda g: box_iou(pred, g))
        pairs.append((pred, best_gt))
    return pairs


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    weights="runs/detect/license_plate_detector/weights/best.pt",
    img_dir="data/val/images",
    ann_dir="data/annotations",
    conf=0.25,
    iou_threshold=0.5,
):
    model = YOLO(weights)

    all_iou, all_ciou = [], []
    tp = fp = fn = 0

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"No images found in {img_dir}")
        return

    for fname in image_files:
        img_path = os.path.join(img_dir, fname)
        stem = os.path.splitext(fname)[0]
        xml_path = os.path.join(ann_dir, stem + ".xml")

        img = Image.open(img_path)
        img_w, img_h = img.size

        # ground truth
        gt_boxes = load_gt_boxes(xml_path, img_w, img_h) if os.path.exists(xml_path) else []

        # predictions
        result = model.predict(img_path, conf=conf, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().tolist()  # [[x1,y1,x2,y2], ...]

        pairs = match_boxes(pred_boxes, gt_boxes)

        matched_gt = set()
        for pred, best_gt in pairs:
            if best_gt is None:
                fp += 1
                continue
            iou  = box_iou(pred, best_gt)
            ciou = box_ciou(pred, best_gt)
            all_iou.append(iou)
            all_ciou.append(ciou)

            gt_idx = gt_boxes.index(best_gt)
            if iou >= iou_threshold and gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_iou  = sum(all_iou)  / len(all_iou)  if all_iou  else 0.0
    mean_ciou = sum(all_ciou) / len(all_ciou) if all_ciou else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Images evaluated : {len(image_files)}")
    print(f"  IoU threshold    : {iou_threshold}")
    print(f"  Mean IoU         : {mean_iou:.4f}")
    print(f"  Mean CIoU        : {mean_ciou:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  TP / FP / FN     : {tp} / {fp} / {fn}")
    print("────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    evaluate()
