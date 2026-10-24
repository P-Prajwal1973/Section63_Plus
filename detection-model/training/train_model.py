"""
train_model.py
---------------
Fine-tunes a pretrained EfficientNet-B0 on the face-crop dataset produced by
detection-model/data_prep/prepare_dataset.py, to classify real vs. AI-generated
(fake) faces.

This is the "hello world" milestone: run this on your TINY sample first
(the ~20 real + ~20 fake videos you processed earlier) just to prove the
whole pipeline works — folder of images -> trained model -> predictions.
Accuracy on this tiny sample won't mean much yet; that's expected. Once
this runs cleanly end-to-end, scale up the dataset size, not the code.

Expects a folder structure like:
    processed_dataset/
        real/
            video1_frame0000.jpg
            ...
        fake/
            video2_frame0000.jpg
            ...

This matches exactly what prepare_dataset.py outputs, and is the standard
layout torchvision.datasets.ImageFolder expects (folder name = class label).

Install requirements first:
    pip install torch torchvision tqdm

Example usage:
    python train_model.py \
        --data_dir ../data_prep/processed_dataset \
        --epochs 5 \
        --batch_size 8 \
        --output_path ./model_checkpoint.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_dataloaders(data_dir, batch_size, image_size=224, val_split=0.2):
    """Load the ImageFolder dataset and split into train/val loaders."""
    # Standard ImageNet normalization — required because we're using an
    # ImageNet-pretrained backbone; the model expects inputs scaled this way.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No images found in {data_dir}. Expected subfolders like real/ and fake/ "
            f"containing .jpg images (output of prepare_dataset.py)."
        )

    print(f"Found {len(full_dataset)} images. Classes: {full_dataset.classes}")
    # full_dataset.classes will be ['fake', 'real'] (alphabetical) —
    # full_dataset.class_to_idx tells you the exact index mapping used below.
    print(f"Class-to-index mapping: {full_dataset.class_to_idx}")

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes=2, device="cpu"):
    """Load a pretrained EfficientNet-B0 and replace its final layer for our 2 classes."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze the pretrained backbone so we only train the new classifier head.
    # This is "transfer learning" — we reuse everything EfficientNet already
    # learned about images in general, and only teach it the new real/fake task.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer (originally 1000 ImageNet classes)
    # with a fresh layer for our 2 classes (real / fake). This new layer's
    # weights start random and ARE trainable.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for real vs. fake face classification")
    parser.add_argument("--data_dir", required=True, help="Path to processed_dataset/ (contains real/ and fake/ subfolders)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (keep small for a tiny dataset)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_path", default="./model_checkpoint.pth", help="Where to save the trained model")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"ERROR: data_dir does not exist: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, classes = build_dataloaders(data_dir, args.batch_size)

    model = build_model(num_classes=len(classes), device=device)
    criterion = nn.CrossEntropyLoss()
    # Only the new classifier head has requires_grad=True, so only it gets trained.
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.epochs} epochs...\n")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
            }, args.output_path)

    print(f"\nDone. Best val_acc={best_val_acc:.3f}. Model saved to {args.output_path}")
    print(
        "\nNote: on a tiny sample dataset (~20-40 videos), don't read much into the "
        "accuracy number itself — the goal of this run is confirming the pipeline "
        "works end-to-end. Re-run on the full dataset once this completes cleanly."
    )


if __name__ == "__main__":
    main()