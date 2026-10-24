"""Run the trained deepfake classifier on an image or a labeled test folder."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, classes


def predict_image(model, classes, image_path: Path, transform):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    index = int(probabilities.argmax().item())
    return classes[index], float(probabilities[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="One image to classify")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Labeled folder containing class subfolders, for example Test/Fake and Test/Real",
    )
    args = parser.parse_args()

    if (args.image is None) == (args.data_dir is None):
        parser.error("provide exactly one of --image or --data_dir")

    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    model, classes = load_model(args.checkpoint)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if args.image is not None:
        if not args.image.is_file():
            parser.error(f"image does not exist: {args.image}")
        label, confidence = predict_image(model, classes, args.image, transform)
        print(f"{args.image}: {label} ({confidence:.1%})")
        return

    image_paths = [
        path
        for path in args.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        parser.error(f"no supported images found under {args.data_dir}")

    correct = 0
    total = 0
    for image_path in sorted(image_paths):
        expected = image_path.parent.name
        predicted, confidence = predict_image(model, classes, image_path, transform)
        correct += predicted.lower() == expected.lower()
        total += 1
        print(
            f"{image_path}: predicted={predicted} "
            f"confidence={confidence:.1%} expected={expected}"
        )

    print(f"\nAccuracy: {correct}/{total} ({correct / total:.1%})")


if __name__ == "__main__":
    main()
