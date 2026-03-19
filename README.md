# License Plate Detection — YOLOv8

## Setup

```bash
pip install ultralytics pillow
```

## Dataset

Download the [Larxel Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) dataset from Kaggle and place files like this:

```
data/
  images/       ← all .png/.jpg images
  annotations/  ← all .xml Pascal VOC annotation files
```

## Usage

```bash
# 1. Convert VOC XML annotations to YOLO format
python convert_annotations.py

# 2. Split into train/val sets
python prepare_dataset.py

# 3. Train
python train.py

# 4. Predict on an image or folder
python predict.py path/to/image.jpg
```

## Notes

- Dataset has 433 images — `augment=True` and pretrained weights help a lot
- Swap `yolov8n.pt` → `yolov8s.pt` or `yolov8m.pt` in `train.py` for better accuracy at the cost of speed
- Best weights saved to `runs/detect/license_plate_detector/weights/best.pt`

CarPlate