from ultralytics import YOLO
from evaluate import evaluate


def main():
    # yolov8n = nano (fastest), swap to yolov8s or yolov8m for better accuracy
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="license_plate_detector",
        patience=20,       # early stopping if no improvement
        augment=True,      # important for small datasets like this one
    )

    print("Training complete. Best weights saved to runs/detect/license_plate_detector/weights/best.pt")

    # run IoU / CIoU evaluation on val set
    evaluate(
        weights="runs/detect/license_plate_detector/weights/best.pt",
        img_dir="data/val/images",
        ann_dir="data/annotations",
        conf=0.25,
        iou_threshold=0.5,
    )


if __name__ == "__main__":
    main()
