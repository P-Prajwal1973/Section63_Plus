"""
extract_features.py
--------------------
Step 1. Turn videos into saved frame embeddings (the slow step - run it once).

Expected folder layout (generator sub-folders inside real/ and fake/ are fine):
    ROOT/
        real/   *.mp4 ...
        fake/   *.mp4 ...   (or fake/<generator_name>/*.mp4)

Output: one small .npy file per video (shape [frames, 512]) under OUT/real and
OUT/fake. Re-running skips videos that are already done, so it is safe to stop
and resume.

Example (start SMALL - 100 videos per class - to check everything works):
    python extract_features.py --root D:/data/genvideo --out features --frames 8 --max_per_class 100
"""
import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import common


def list_videos(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in common.VIDEO_EXTS)


def main():
    ap = argparse.ArgumentParser(description="Extract CLIP frame embeddings from videos")
    ap.add_argument("--root", required=True, help="Folder containing real/ and fake/")
    ap.add_argument("--out", required=True, help="Where to save the .npy features")
    ap.add_argument("--frames", type=int, default=8, help="Frames sampled per video")
    ap.add_argument("--max_per_class", type=int, default=200, help="Max videos per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    device = common.get_device()
    print(f"Using device: {device}")
    model, processor = common.load_backbone(device)
    rng = random.Random(args.seed)

    for label in ("real", "fake"):
        folder = root / label
        if not folder.exists():
            print(f"[warn] {folder} not found - skipping '{label}'")
            continue
        videos = list_videos(folder)
        rng.shuffle(videos)
        videos = videos[: args.max_per_class]
        (out / label).mkdir(parents=True, exist_ok=True)

        ok_count = failed = 0
        for v in tqdm(videos, desc=label):
            name = "__".join(v.relative_to(folder).with_suffix("").parts) + ".npy"
            dest = out / label / name
            if dest.exists():
                ok_count += 1
                continue
            try:
                frames = common.sample_frames(v, args.frames)
                if len(frames) < 2:
                    raise ValueError("fewer than 2 readable frames")
                emb = common.embed_frames(frames, model, processor, device)
                np.save(dest, emb)
                ok_count += 1
            except Exception as e:  # keep going on a bad/corrupt video
                failed += 1
                print(f"  [skip] {v.name}: {e}")
        print(f"{label}: {ok_count} ready, {failed} skipped (found {len(videos)} videos)")


if __name__ == "__main__":
    main()
