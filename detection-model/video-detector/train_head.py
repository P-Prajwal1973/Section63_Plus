"""
train_head.py
--------------
Step 2. Train the small classifier on the saved embeddings (takes seconds).

IMPORTANT: the train/validation split is done PER VIDEO, never per frame, so
frames of the same video can't leak into both sets.

The validation score only tells you the model learned the generators it saw.
The honest test is --test_features: features extracted (with extract_features.py)
from a DIFFERENT generator or dataset that was never used for training.

Example:
    python train_head.py --features features --out head.pt
    python train_head.py --features features --test_features features_other --out head.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import common


def load_features(feat_dir):
    X, y, lengths = [], [], []
    for label, cls in (("real", 0), ("fake", 1)):
        for f in sorted((Path(feat_dir) / label).glob("*.npy")):
            emb = np.load(f)
            X.append(common.pool_features(emb))
            y.append(cls)
            lengths.append(len(emb))
    if not X:
        raise SystemExit(f"No .npy features found in {feat_dir}/real and /fake")
    return np.stack(X), np.array(y), int(np.median(lengths))


def report(title, y_true, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    print(f"\n=== {title} ({len(y_true)} videos) ===")
    print(f"accuracy={acc:.3f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    print(f"false-positive rate (real flagged as fake)={fpr:.3f}")
    print(f"confusion matrix  [[TN={tn} FP={fp}] [FN={fn} TP={tp}]]")


def predict_probs(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def main():
    ap = argparse.ArgumentParser(description="Train the classifier head on saved features")
    ap.add_argument("--features", required=True, help="Folder from extract_features.py")
    ap.add_argument("--test_features", default=None, help="Optional features from an UNSEEN generator/dataset")
    ap.add_argument("--out", default="head.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--patience", type=int, default=20, help="Stop if val loss doesn't improve")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    X, y, n_frames = load_features(args.features)
    print(f"Loaded {len(y)} videos ({int((y == 0).sum())} real, {int((y == 1).sum())} fake), "
          f"feature size {X.shape[1]}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=args.seed)

    model = common.Head(X.shape[1], args.hidden)
    pos_weight = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xva_t), yva_t).item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | val_loss={val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch} (best val_loss={best:.4f})")
                break

    model.load_state_dict(best_state)
    report("Validation (same generators as training)", yva, predict_probs(model, Xva))

    if args.test_features:
        Xte, yte, _ = load_features(args.test_features)
        report("UNSEEN test set (cross-generator / cross-dataset)", yte, predict_probs(model, Xte))

    torch.save({"state_dict": model.state_dict(), "in_dim": X.shape[1], "hidden": args.hidden,
                "model_name": common.MODEL_NAME, "n_frames": n_frames}, args.out)
    print(f"\nSaved classifier to {args.out}")


if __name__ == "__main__":
    main()
