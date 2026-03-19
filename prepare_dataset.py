import os
import shutil
import random


def split(img_dir, lbl_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print(f"No images found in {img_dir}")
        return

    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split_name, files in splits.items():
        for subdir in ["images", "labels"]:
            os.makedirs(f"data/{split_name}/{subdir}", exist_ok=True)
        for fname in files:
            stem = os.path.splitext(fname)[0]
            shutil.copy(os.path.join(img_dir, fname), f"data/{split_name}/images/{fname}")
            lbl = stem + ".txt"
            lbl_src = os.path.join(lbl_dir, lbl)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, f"data/{split_name}/labels/{lbl}")
            else:
                print(f"Warning: label not found for {fname}")

    print(f"Split complete — train: {len(splits['train'])}, val: {len(splits['val'])}")


if __name__ == "__main__":
    split("data/images", "data/labels")
