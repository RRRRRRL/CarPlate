import argparse
import os
import random
import shutil
import yaml


def split_data(image_dir, label_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)

    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    random.shuffle(images)
    train_count = int(len(images) * split_ratio)
    splits = {"train": images[:train_count], "val": images[train_count:]}

    for split_name, files in splits.items():
        for image_name in files:
            image_src = os.path.join(image_dir, image_name)
            image_dst = os.path.join(output_dir, "images", split_name, image_name)
            shutil.copy2(image_src, image_dst)

            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_src = os.path.join(label_dir, label_name)
            if os.path.exists(label_src):
                label_dst = os.path.join(output_dir, "labels", split_name, label_name)
                shutil.copy2(label_src, label_dst)

    yaml_data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["car_plate"],
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print(f"Data split complete: train={len(splits['train'])}, val={len(splits['val'])}")
    print(f"YOLO data config written to {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val and generate data.yaml")
    parser.add_argument("--image-dir", default="data/images", help="Path to source images")
    parser.add_argument("--label-dir", default="data/labels", help="Path to YOLO txt labels")
    parser.add_argument("--output-dir", default="dataset", help="Output dataset root")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_data(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
