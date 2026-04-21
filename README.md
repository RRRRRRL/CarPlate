# Car Plate Detection — Official YOLOv8 vs Improved Custom YOLO

This repository now includes a complete pipeline to compare the official YOLOv8 model with a custom YOLO architecture designed for car plate detection.

## What was added

- `yolov8_model.py`: Loads the official YOLOv8 baseline model.
- `custom_yolo_model.py`: `ImprovedCustomYOLO` architecture with:
  - C2f blocks
  - SPPF
  - Top-Down FPN neck
  - Decoupled multi-scale heads at stride 8, 16, and 32
- `prepare_data.py`: Splits image/label data into 80/20 train/val and writes `dataset/data.yaml`.
- `train_yolov8.py`: Trains the official YOLOv8 model using Ultralytics and saves checkpoints.
- `train_custom.py`: Structural PyTorch training loop for the custom model and saves best checkpoint.
- `compare.py`: Compares model size, inference time, and FPS.

## Repository structure

- Core comparison pipeline files:
  - `yolov8_model.py` — official model loader
  - `custom_yolo_model.py` — custom architecture
  - `prepare_data.py` — 80/20 split + `dataset/data.yaml` generator
  - `train_yolov8.py` — official YOLOv8 training script
  - `train_custom.py` — custom model training loop
  - `compare.py` — model comparison script
- Supporting utilities:
  - `convert_annotations.py` — VOC XML to YOLO txt labels
  - `evaluate.py` — IoU/CIoU evaluation
  - `predict.py` — prediction script

## Installation

```bash
pip install torch torchvision ultralytics pillow pyyaml
```

## Usage pipeline

### 1) Convert annotations (if starting from VOC XML)

```bash
python convert_annotations.py
```

### 2) Prepare train/val split and generate `data.yaml`

```bash
python prepare_data.py --image-dir data/images --label-dir data/labels --output-dir dataset
```

This creates:

- `dataset/images/train`, `dataset/images/val`
- `dataset/labels/train`, `dataset/labels/val`
- `dataset/data.yaml`

### 3) Train official YOLOv8 baseline

```bash
python train_yolov8.py --data dataset/data.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

Best checkpoint path:

- `runs/official_yolo/car_plate_model/weights/best.pt`

### 4) Train custom model

```bash
python train_custom.py --epochs 50 --batch-size 8 --save-path runs/custom_yolo/custom_yolo_best.pth
```

Best checkpoint path:

- `runs/custom_yolo/custom_yolo_best.pth`

### 5) Compare models

```bash
python compare.py \
  --official-weights runs/official_yolo/car_plate_model/weights/best.pt \
  --custom-weights runs/custom_yolo/custom_yolo_best.pth \
  --runs 50 --img-size 640
```

## Evaluation metrics reported by `compare.py`

- **Model Size (MB)**
- **Inference Time (ms)**
- **Estimated FPS**

## Notes

- `train_custom.py` is intentionally a structural training loop; replace placeholder data/target assignment with your dataset loader and detection losses for production-quality results.
- Keep image/label filenames aligned (`image_001.jpg` ↔ `image_001.txt`) before running data preparation.
