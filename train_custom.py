import argparse
import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from custom_yolo_model import ImprovedCustomYOLO

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
RESAMPLE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


class LarxelDataset(Dataset):
    """Larxel-format dataset reader (images/<split>, labels/<split>)."""

    def __init__(self, dataset_dir="dataset", split="train", img_size=640):
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.image_dir = self.dataset_dir / "images" / split
        self.label_dir = self.dataset_dir / "labels" / split

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.image_files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS])
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size), RESAMPLE_BILINEAR)
        image_tensor = TF.to_tensor(image)

        labels = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id, x, y, w, h = map(float, parts[:5])
                    labels.append([cls_id, x, y, w, h])

        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        return image_tensor, label_tensor


def collate_fn(batch):
    # Keeps labels as a list so each image can carry a variable number of bounding boxes.
    images = torch.stack([sample[0] for sample in batch], dim=0)
    labels = [sample[1] for sample in batch]
    return images, labels


def build_targets_for_scale(batch_labels, pred_cls, device):
    batch_size, _, grid_h, grid_w = pred_cls.shape
    target_cls = torch.zeros((batch_size, 1, grid_h, grid_w), device=device)
    target_box = torch.zeros((batch_size, 4, grid_h, grid_w), device=device)
    target_count = torch.zeros((batch_size, 1, grid_h, grid_w), device=device)

    for b_idx, labels in enumerate(batch_labels):
        if labels.numel() == 0:
            continue
        labels = labels.to(device)

        for label in labels:
            x, y, w, h = label[1], label[2], label[3], label[4]
            gx = min(max(int(x.item() * grid_w), 0), grid_w - 1)
            gy = min(max(int(y.item() * grid_h), 0), grid_h - 1)

            target_cls[b_idx, 0, gy, gx] = 1.0
            target_box[b_idx, 0, gy, gx] += x * grid_w - gx
            target_box[b_idx, 1, gy, gx] += y * grid_h - gy
            target_box[b_idx, 2, gy, gx] += w
            target_box[b_idx, 3, gy, gx] += h
            target_count[b_idx, 0, gy, gx] += 1.0

    target_box = target_box / target_count.clamp(min=1.0)

    return target_cls, target_box


def train_custom_model(
    epochs=50,
    batch_size=8,
    lr=1e-3,
    save_path="runs/custom_yolo/custom_yolo_best.pth",
    dataset_dir="dataset",
    img_size=640,
    num_workers=2,
):
    """Train ImprovedCustomYOLO with Larxel-format real data.

    Args:
        epochs: Number of epochs to run.
        batch_size: Batch size for DataLoader.
        lr: Optimizer learning rate.
        save_path: Destination path for best checkpoint.
        dataset_dir: Dataset root with images/labels train/val folders.
        img_size: Input image size (square).
        num_workers: DataLoader workers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCustomYOLO(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    cls_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.MSELoss()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    train_dataset = LarxelDataset(dataset_dir=dataset_dir, split="train", img_size=img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    best_loss = float("inf")
    model.train()
    print(f"Training custom model on {device}")
    print(f"Loaded {len(train_dataset)} training images from {train_dataset.image_dir}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        processed_batches = 0

        for images, batch_labels in train_loader:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad()
            (cls_s, box_s), (cls_m, box_m), (cls_l, box_l) = model(images)

            target_cls_s, target_box_s = build_targets_for_scale(batch_labels, cls_s, device)
            target_cls_m, target_box_m = build_targets_for_scale(batch_labels, cls_m, device)
            target_cls_l, target_box_l = build_targets_for_scale(batch_labels, cls_l, device)

            loss = (
                cls_loss_fn(cls_s, target_cls_s)
                + box_loss_fn(box_s, target_box_s)
                + cls_loss_fn(cls_m, target_cls_m)
                + box_loss_fn(box_m, target_box_m)
                + cls_loss_fn(cls_l, target_cls_l)
                + box_loss_fn(box_l, target_box_l)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_batches += 1

        if processed_batches == 0:
            raise RuntimeError("No training batches were processed. Check dataset and DataLoader settings.")

        avg_loss = epoch_loss / processed_batches
        print(f"Epoch [{epoch + 1}/{epochs}] loss={avg_loss:.4f} (batches={processed_batches})")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best custom checkpoint to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom YOLO architecture")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", default="runs/custom_yolo/custom_yolo_best.pth")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        dataset_dir=args.dataset_dir,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
