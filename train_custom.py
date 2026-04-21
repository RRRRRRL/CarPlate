import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from custom_yolo_model import ImprovedCustomYOLO


def train_custom_model(epochs=50, batch_size=8, lr=1e-3, save_path="runs/custom_yolo/custom_yolo_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCustomYOLO(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    cls_loss_fn = nn.BCEWithLogitsLoss()
    box_loss_fn = nn.MSELoss()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_loss = float("inf")
    model.train()
    print(f"Training custom model on {device}")

    # Structural training loop with placeholder data/targets.
    # Replace this section with a dataset loader + detection target assignment.
    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(10):
            images = torch.randn(batch_size, 3, 640, 640, device=device)

            optimizer.zero_grad()
            (cls_s, box_s), _, _ = model(images)

            target_cls = torch.zeros_like(cls_s)
            target_box = torch.zeros_like(box_s)

            loss = cls_loss_fn(cls_s, target_cls) + box_loss_fn(box_s, target_box)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / 10
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )
