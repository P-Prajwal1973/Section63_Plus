"""
prepare_dataset.py
-------------------
Frame extraction + face cropping pipeline for FaceForensics++ (C23).

What this does:
  1. Walks a folder of videos (real or fake).
  2. Extracts N evenly-spaced frames per video.
  3. Detects and crops the face in each frame using MTCNN.
  4. Saves cropped face images into an output folder, organized as:
         output_dir/real/<video_name>_frame0001.jpg
         output_dir/fake/<video_name>_frame0001.jpg

Why this structure:
  This is the exact folder layout most PyTorch tutorials expect for
  torchvision.datasets.ImageFolder — so once this script has run, you
  can point a standard training script straight at output_dir with no
  extra work.

First run (recommended): start SMALL. Use --max_videos 20 to build a
tiny sanity-check dataset before running this on the full FF++ set.
That's the "hello world" step — get the whole pipeline working on 20
videos before committing to hours of processing on the full dataset.

Install requirements first:
    pip install facenet-pytorch opencv-python torch torchvision tqdm

Example usage:
    # Process 20 real videos
    python prepare_dataset.py \
        --input_dir /path/to/FaceForensics++_C23/original_sequences/youtube/c23/videos \
        --output_dir ./processed_dataset \
        --label real \
        --max_videos 20 \
        --frames_per_video 10

    # Process 20 fake videos (Deepfakes method)
    python prepare_dataset.py \
        --input_dir /path/to/FaceForensics++_C23/manipulated_sequences/Deepfakes/c23/videos \
        --output_dir ./processed_dataset \
        --label fake \
        --max_videos 20 \
        --frames_per_video 10
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm


def get_video_files(input_dir):
    """Find all video files in a directory (mp4/avi)."""
    exts = {".mp4", ".avi", ".mov"}
    return sorted(
        [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]
    )


def extract_evenly_spaced_frames(video_path, n_frames):
    """Read a video and return n_frames evenly spaced frames (as BGR numpy arrays)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    # Evenly spaced frame indices across the whole video
    n_frames = min(n_frames, total)
    indices = sorted(set(int(i * total / n_frames) for i in range(n_frames)))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames + crop faces from FaceForensics++ videos")
    parser.add_argument("--input_dir", required=True, help="Folder containing .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Where to save cropped face images")
    parser.add_argument("--label", required=True, choices=["real", "fake"], help="Label for this batch of videos")
    parser.add_argument("--max_videos", type=int, default=20, help="Max number of videos to process (start small!)")
    parser.add_argument("--frames_per_video", type=int, default=10, help="Frames to sample per video")
    parser.add_argument("--face_size", type=int, default=224, help="Output face crop size (pixels, square)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        sys.exit(f"ERROR: input_dir does not exist: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # MTCNN handles detection + alignment + cropping in one call.
    # image_size is the output crop size; margin adds context around the face.
    mtcnn = MTCNN(image_size=args.face_size, margin=20, keep_all=False, device=device)

    videos = get_video_files(input_dir)[: args.max_videos]
    if not videos:
        sys.exit(f"ERROR: no video files found in {input_dir}")

    print(f"Found {len(videos)} videos. Processing (label={args.label})...")

    total_faces_saved = 0
    total_faces_missed = 0

    for video_path in tqdm(videos, desc="Videos"):
        frames = extract_evenly_spaced_frames(video_path, args.frames_per_video)

        for i, frame_bgr in enumerate(frames):
            # MTCNN expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            save_path = output_dir / f"{video_path.stem}_frame{i:04d}.jpg"

            try:
                # mtcnn(...) with save_path writes the cropped face directly to disk.
                # Returns None if no face was detected in this frame.
                face_tensor = mtcnn(frame_rgb, save_path=str(save_path))
                if face_tensor is not None:
                    total_faces_saved += 1
                else:
                    total_faces_missed += 1
            except Exception as e:
                print(f"  [warn] failed on {video_path.name} frame {i}: {e}")
                total_faces_missed += 1

    print("\nDone.")
    print(f"  Faces saved:  {total_faces_saved}")
    print(f"  Frames with no face detected (skipped): {total_faces_missed}")
    print(f"  Output folder: {output_dir}")


if __name__ == "__main__":
    main()
