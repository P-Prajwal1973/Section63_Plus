"""
common.py
---------
Shared helpers for the whole-frame AI-generated video detector.

Approach (no face detection, works on any video):
  1. Sample N evenly spaced frames from the video.
  2. Turn each frame into an embedding with a FROZEN pretrained CLIP vision
     model (we never train CLIP itself - that is what keeps this feasible
     on a laptop).
  3. Pool the frame embeddings into one video-level vector:
       mean   -> what the video looks like overall
       std    -> how much the content varies across frames
       diff   -> frame-to-frame change (a cheap temporal-consistency signal)
  4. Train a small classifier head on those pooled vectors.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

MODEL_NAME = "openai/clip-vit-base-patch32"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(device):
    """Load the frozen CLIP vision encoder (downloads ~600 MB on first run)."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def sample_frames(video_path, n_frames):
    """Return up to n_frames evenly spaced RGB frames (numpy HxWx3 uint8)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:
        cap.release()
        return frames
    idxs = np.linspace(0, total - 1, num=min(n_frames, total)).astype(int)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@torch.no_grad()
def embed_frames(frames, model, processor, device, batch_size=16):
    """Frames -> L2-normalised CLIP embeddings, shape [T, 512] (numpy)."""
    outs = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        emb = model(pixel_values=inputs["pixel_values"].to(device)).image_embeds
        emb = emb / emb.norm(dim=-1, keepdim=True)
        outs.append(emb.cpu())
    return torch.cat(outs).numpy()


def pool_features(emb):
    """[T, D] frame embeddings -> one [3*D] video vector (mean, std, mean abs diff)."""
    mean = emb.mean(axis=0)
    if len(emb) > 1:
        std = emb.std(axis=0)
        diff = np.abs(np.diff(emb, axis=0)).mean(axis=0)
    else:
        std = np.zeros_like(mean)
        diff = np.zeros_like(mean)
    return np.concatenate([mean, std, diff]).astype(np.float32)


class Head(nn.Module):
    """Small classifier on top of pooled video features. Output = logit (fake)."""

    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
