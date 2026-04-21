from ultralytics import YOLO


def get_yolov8_model(weights_path: str = "yolov8n.pt") -> YOLO:
    """Load the official YOLOv8 model from Ultralytics."""
    return YOLO(weights_path)


if __name__ == "__main__":
    model = get_yolov8_model()
    print("YOLOv8 model loaded successfully")
    model.info()
