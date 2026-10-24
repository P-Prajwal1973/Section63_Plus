"""
predict.py
-----------
Step 3. Score a new video. Works for ANY video - no face needed.

CLI:
    python predict.py --video suspect.mp4 --checkpoint head.pt

From other code (e.g. the FastAPI backend):
    from predict import VideoDetector
    detector = VideoDetector("head.pt")      # load once at server start
    result = detector.predict("suspect.mp4")  # dict with probability + risk band
"""
import argparse
import json

import numpy as np
import torch

import common


def risk_band(p):
    # PLACEHOLDER thresholds - calibrate on your validation/unseen-test results
    # before trusting these bands in the certificate annexure.
    if p < 0.35:
        return "Low"
    if p < 0.65:
        return "Medium"
    return "High"


class VideoDetector:
    def __init__(self, checkpoint, device=None):
        self.device = device or common.get_device()
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.n_frames = ckpt["n_frames"]
        self.head = common.Head(ckpt["in_dim"], ckpt["hidden"])
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        self.model, self.processor = common.load_backbone(self.device)

    def predict(self, video_path):
        frames = common.sample_frames(video_path, self.n_frames)
        if len(frames) < 2:
            raise ValueError("Could not read at least 2 frames from this video")
        emb = common.embed_frames(frames, self.model, self.processor, self.device)
        x = torch.tensor(common.pool_features(emb)).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(self.head(x)).item()
        return {
            "video": str(video_path),
            "fake_probability": round(prob, 4),
            "risk_band": risk_band(prob),
            "frames_analysed": len(frames),
            "backbone": common.MODEL_NAME,
        }


def main():
    ap = argparse.ArgumentParser(description="Score a video for AI generation")
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="head.pt")
    args = ap.parse_args()
    print(json.dumps(VideoDetector(args.checkpoint).predict(args.video), indent=2))


if __name__ == "__main__":
    main()
