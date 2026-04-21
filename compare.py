import argparse
import os
import time

import numpy as np
import torch
from ultralytics import YOLO

from custom_yolo_model import ImprovedCustomYOLO


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


def compare_models(official_weights, custom_weights, runs=50, img_size=640):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    official_model = YOLO(official_weights)
    custom_model = ImprovedCustomYOLO(num_classes=1).to(device)
    custom_model.load_state_dict(torch.load(custom_weights, map_location=device))
    custom_model.eval()

    official_size = get_model_size_mb(official_weights)
    custom_size = get_model_size_mb(custom_weights)

    official_time = benchmark_official(official_model, runs=runs, img_size=img_size)
    custom_time = benchmark_custom(custom_model, device=device, runs=runs, img_size=img_size)

    print("\n" + "=" * 64)
    print("YOLOv8 Official vs ImprovedCustomYOLO")
    print("=" * 64)
    print(f"{'Metric':<24}{'Official YOLOv8':<20}{'ImprovedCustomYOLO':<20}")
    print("-" * 64)
    print(f"{'Model Size (MB)':<24}{official_size:<20.2f}{custom_size:<20.2f}")
    print(f"{'Inference Time (ms)':<24}{official_time * 1000:<20.2f}{custom_time * 1000:<20.2f}")
    print(f"{'Estimated FPS':<24}{(1 / official_time):<20.2f}{(1 / custom_time):<20.2f}")
    print("=" * 64)


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
    args = parser.parse_args()

    compare_models(
        official_weights=args.official_weights,
        custom_weights=args.custom_weights,
        runs=args.runs,
        img_size=args.img_size,
    )
