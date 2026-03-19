import sys
from ultralytics import YOLO


def main():
    weights = "runs/detect/license_plate_detector/weights/best.pt"
    source = sys.argv[1] if len(sys.argv) > 1 else "data/val/images"

    model = YOLO(weights)
    results = model.predict(source=source, save=True, conf=0.25)

    for r in results:
        print(r.boxes)


if __name__ == "__main__":
    main()
