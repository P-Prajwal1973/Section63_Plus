# Dataset Preparation — FaceForensics++ (C23)

This is the "hello world" step for Prajwal's track: turn raw FF++ videos into
a folder of cropped face images ready for model fine-tuning.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, make sure you install the CUDA-enabled build of
PyTorch (see https://pytorch.org/get-started/locally/ for the right command
for your system) — this script will automatically use the GPU if available,
and CPU-only face detection is noticeably slower.

## 2. Run on a SMALL subset first

Don't point this at the full dataset on your first run. Start with ~20 real
and ~20 fake videos to confirm the whole pipeline works end-to-end:

```bash
# Real videos
python prepare_dataset.py \
    --input_dir /path/to/FaceForensics++_C23/original_sequences/youtube/c23/videos \
    --output_dir ./processed_dataset \
    --label real \
    --max_videos 20 \
    --frames_per_video 10

# Fake videos (Deepfakes method — pick one method to start)
python prepare_dataset.py \
    --input_dir /path/to/FaceForensics++_C23/manipulated_sequences/Deepfakes/c23/videos \
    --output_dir ./processed_dataset \
    --label fake \
    --max_videos 20 \
    --frames_per_video 10
```

After this you should have:

```
processed_dataset/
    real/
        video1_frame0000.jpg
        video1_frame0001.jpg
        ...
    fake/
        video2_frame0000.jpg
        ...
```

This layout works directly with `torchvision.datasets.ImageFolder`, so your
training script can load it with almost no extra code.

## 3. Sanity check before scaling up

- Open a handful of the saved face crops and visually confirm they actually
  contain cropped, centered faces (not blank/black images — that usually
  means MTCNN failed to detect a face and something's off with input format).
- Check the printed "Faces saved" vs "Frames with no face detected" counts —
  if a large fraction of frames have no detected face, the video quality or
  face angle may need a lower detection threshold (see facenet-pytorch's
  MTCNN docs for the `thresholds` parameter).

## 4. Scale up

Once the small run looks correct, increase `--max_videos` and
`--frames_per_video`, and repeat for the other manipulation methods
(Face2Face, FaceSwap, NeuralTextures) if you want a more diverse fake set —
FaceForensics++ ships all four, and using more than one method generally
improves how well your detector generalizes.

## Notes

- Videos with very few frames or corrupted files are skipped automatically
  with a warning printed to the console — check these warnings, don't just
  ignore them.
- `--frames_per_video` controls how many frames are sampled evenly across
  each video's duration — 10 is a reasonable starting point; you can push
  this higher once you know your storage budget.
- Face crops are square, sized by `--face_size` (default 224px, which
  matches the typical input size for pretrained CNN backbones like
  XceptionNet/EfficientNet — no resizing needed later).
