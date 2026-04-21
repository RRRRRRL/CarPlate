import argparse
from ultralytics import YOLO


def train_official_yolo(data_yaml="dataset/data.yaml", weights="yolov8n.pt", epochs=50, imgsz=640, batch=16):
    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/official_yolo",
        name="car_plate_model",
    )
    print("Training complete. Best weights: runs/official_yolo/car_plate_model/weights/best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train official Ultralytics YOLOv8 baseline")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_official_yolo(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
