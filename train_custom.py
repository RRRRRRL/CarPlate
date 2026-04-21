import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from custom_yolo_model import ImprovedCustomYOLO


def train_custom_model(
    epochs=50,
    batch_size=8,
    lr=1e-3,
    save_path="runs/custom_yolo/custom_yolo_best.pth",
    batches_per_epoch=10,
):
    """Train ImprovedCustomYOLO using a structural placeholder loop.

    Args:
        epochs: Number of epochs to run.
        batch_size: Batch size for dummy inputs.
        lr: Optimizer learning rate.
        save_path: Destination path for best checkpoint.

    Notes:
        This loop intentionally uses random tensors as placeholder data. Replace
        it with a dataset loader and detection target assignment for real
        training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCustomYOLO(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    cls_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_loss = float("inf")
    model.train()
    print(f"Training custom model on {device}")
    print("Warning: this script uses placeholder synthetic data/targets for structural training only.")

    # Structural training loop with placeholder data/targets.
    # Replace this section with a dataset loader + detection target assignment.
    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(batches_per_epoch):
            images = torch.randn(batch_size, 3, 640, 640, device=device)

            optimizer.zero_grad()
            (cls_s, box_s), (cls_m, box_m), (cls_l, box_l) = model(images)

            target_cls_s = torch.zeros_like(cls_s)
            target_box_s = torch.zeros_like(box_s)
            target_cls_m = torch.zeros_like(cls_m)
            target_box_m = torch.zeros_like(box_m)
            target_cls_l = torch.zeros_like(cls_l)
            target_box_l = torch.zeros_like(box_l)

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

        avg_loss = epoch_loss / batches_per_epoch
        print(f"Epoch [{epoch + 1}/{epochs}] loss={avg_loss:.4f}")

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
    parser.add_argument("--batches-per-epoch", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        batches_per_epoch=args.batches_per_epoch,
    )
