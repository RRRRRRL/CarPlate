import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from custom_yolo_model import ImprovedCustomYOLO

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def get_model_size_mb(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)


def benchmark_official(model, runs=50, img_size=640):
    image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    start = time.perf_counter()
    for _ in range(runs):
        model.predict(source=image, verbose=False)
    avg_seconds = (time.perf_counter() - start) / runs
    return avg_seconds


def benchmark_custom(model, device, runs=50, img_size=640):
    x = torch.randn(1, 3, img_size, img_size, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    avg_seconds = (time.perf_counter() - start) / runs
    return avg_seconds


def yolo_to_xyxy(x, y, w, h, img_size):
    x1 = (x - w / 2.0) * img_size
    y1 = (y - h / 2.0) * img_size
    x2 = (x + w / 2.0) * img_size
    y2 = (y + h / 2.0) * img_size
    return [x1, y1, x2, y2]


def box_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def box_ciou(box1, box2):
    iou = box_iou(box1, box2)

    cx1, cy1 = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
    cx2, cy2 = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    w1, h1 = max(box1[2] - box1[0], 1e-7), max(box1[3] - box1[1], 1e-7)
    w2, h2 = max(box2[2] - box2[0], 1e-7), max(box2[3] - box2[1], 1e-7)
    v = (4.0 / np.pi**2) * (np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2
    alpha = v / (1.0 - iou + v + 1e-7)

    return float(iou - rho2 / c2 - alpha * v)


def load_gt_boxes(label_path, img_size):
    if not label_path.exists():
        return []
    gt_boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x, y, w, h = map(float, parts[:5])
            gt_boxes.append(yolo_to_xyxy(x, y, w, h, img_size))
    return gt_boxes


def decode_custom_outputs(outputs, img_size):
    decoded_boxes = []
    for cls_map, box_map in outputs:
        cls_scores = torch.sigmoid(cls_map[0, 0])
        flat_idx = torch.argmax(cls_scores).item()
        grid_h, grid_w = cls_scores.shape
        gy, gx = divmod(flat_idx, grid_w)

        offsets = torch.sigmoid(box_map[0, :, gy, gx]).tolist()
        cx = ((gx + offsets[0]) / grid_w) * img_size
        cy = ((gy + offsets[1]) / grid_h) * img_size
        bw = offsets[2] * img_size
        bh = offsets[3] * img_size

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        decoded_boxes.append([x1, y1, x2, y2])
    return decoded_boxes


def mean_ciou_official(model, image_files, labels_dir, img_size):
    ciou_scores = []
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = load_gt_boxes(label_path, img_size)
        if not gt_boxes:
            continue

        result = model.predict(source=str(image_path), imgsz=img_size, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        if not pred_boxes:
            continue

        for pred in pred_boxes:
            best_gt = max(gt_boxes, key=lambda gt: box_iou(pred, gt))
            ciou_scores.append(box_ciou(pred, best_gt))
    return float(np.mean(ciou_scores)) if ciou_scores else 0.0


def mean_ciou_custom(model, device, image_files, labels_dir, img_size):
    ciou_scores = []
    with torch.no_grad():
        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_boxes = load_gt_boxes(label_path, img_size)
            if not gt_boxes:
                continue

            image = Image.open(image_path).convert("RGB").resize((img_size, img_size), RESAMPLE_BILINEAR)
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            image_tensor = image_tensor.to(device)

            outputs = model(image_tensor)
            pred_boxes = decode_custom_outputs(outputs, img_size)
            for pred in pred_boxes:
                best_gt = max(gt_boxes, key=lambda gt: box_iou(pred, gt))
                ciou_scores.append(box_ciou(pred, best_gt))
    return float(np.mean(ciou_scores)) if ciou_scores else 0.0


def export_comparison_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Official YOLOv8", "ImprovedCustomYOLO"])
        writer.writerows(rows)


def compare_models(official_weights, custom_weights, runs=50, img_size=640, dataset_dir="dataset", output_file="runs/comparison/performance_comparison.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    official_model = YOLO(official_weights)
    custom_model = ImprovedCustomYOLO(num_classes=1).to(device)
    custom_model.load_state_dict(torch.load(custom_weights, map_location=device))
    custom_model.eval()

    official_size = get_model_size_mb(official_weights)
    custom_size = get_model_size_mb(custom_weights)

    official_time = benchmark_official(official_model, runs=runs, img_size=img_size)
    custom_time = benchmark_custom(custom_model, device=device, runs=runs, img_size=img_size)
    val_images_dir = Path(dataset_dir) / "images" / "val"
    val_labels_dir = Path(dataset_dir) / "labels" / "val"
    image_files = (
        sorted([p for p in val_images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]) if val_images_dir.exists() else []
    )
    official_ciou = mean_ciou_official(official_model, image_files, val_labels_dir, img_size) if image_files else 0.0
    custom_ciou = mean_ciou_custom(custom_model, device, image_files, val_labels_dir, img_size) if image_files else 0.0

    print("\n" + "=" * 64)
    print("YOLOv8 Official vs ImprovedCustomYOLO")
    print("=" * 64)
    print(f"{'Metric':<24}{'Official YOLOv8':<20}{'ImprovedCustomYOLO':<20}")
    print("-" * 64)
    print(f"{'Model Size (MB)':<24}{official_size:<20.2f}{custom_size:<20.2f}")
    print(f"{'Inference Time (ms)':<24}{official_time * 1000:<20.2f}{custom_time * 1000:<20.2f}")
    print(f"{'Estimated FPS':<24}{(1 / official_time):<20.2f}{(1 / custom_time):<20.2f}")
    print(f"{'Mean CIoU':<24}{official_ciou:<20.4f}{custom_ciou:<20.4f}")
    print("=" * 64)

    table_rows = [
        ["Model Size (MB)", f"{official_size:.2f}", f"{custom_size:.2f}"],
        ["Inference Time (ms)", f"{official_time * 1000:.2f}", f"{custom_time * 1000:.2f}"],
        ["Estimated FPS", f"{1 / official_time:.2f}", f"{1 / custom_time:.2f}"],
        ["Mean CIoU", f"{official_ciou:.4f}", f"{custom_ciou:.4f}"],
    ]
    export_comparison_table(table_rows, output_file)
    print(f"Saved comparison table to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare official YOLOv8 and custom YOLO model")
    parser.add_argument(
        "--official-weights",
        default="runs/official_yolo/car_plate_model/weights/best.pt",
        help="Path to official YOLOv8 checkpoint",
    )
    parser.add_argument(
        "--custom-weights",
        default="runs/custom_yolo/custom_yolo_best.pth",
        help="Path to custom model checkpoint",
    )
    parser.add_argument("--runs", type=int, default=50, help="Number of iterations for timing")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size for timing")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root for CIoU evaluation")
    parser.add_argument(
        "--output-file",
        default="runs/comparison/performance_comparison.csv",
        help="Output CSV/TXT file for the final comparison table",
    )
    args = parser.parse_args()

    compare_models(
        official_weights=args.official_weights,
        custom_weights=args.custom_weights,
        runs=args.runs,
        img_size=args.img_size,
        dataset_dir=args.dataset_dir,
        output_file=args.output_file,
    )
