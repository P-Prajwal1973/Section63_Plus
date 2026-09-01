# Section 63+

AI-based detection of AI-generated/tampered video and document evidence, with automated forensic certification aligned to Section 63 of India's Bharatiya Sakshya Adhiniyam (BSA), 2023.

## What this is

A forensic pre-certification tool for cyber-cell investigators. Before electronic evidence (video, image, or document) goes into a Section 63(4) certificate for court, this tool:

1. **Detects** whether video/image evidence is AI-generated or manipulated (deepfake detection), with a secondary tamper-check for documents/static images (Error Level Analysis).
2. **Explains** its verdict visually (Grad-CAM heatmaps) rather than returning a black-box score.
3. **Logs** every step of its own analysis into a tamper-evident, hash-chained record.
4. **Drafts** the Section 63(4) certificate automatically, with an added "AI-Generation Risk Annexure" that the current certificate format doesn't include.

Full project synopsis (introduction, literature survey, methodology, objectives, references) is in [`/docs`](./docs).

## Team

| Member | Focus |
|---|---|
| P Prajwal | System architecture, video/image deepfake detection model |
| Sanskrut C Kodabagi | Chain-of-custody logging, Section 63 certificate mapping |
| Monish D N | Document/image tamper detection (ELA), dataset pipeline |
| N Bhushan | Backend integration, dashboard |

## Repository structure

```
section63-plus/
├── detection-model/      # Video/image deepfake detection (Prajwal)
│   ├── data_prep/          # Frame extraction + face cropping scripts
│   ├── training/            # Model fine-tuning scripts
│   └── inference/           # Run trained model on new evidence
├── document-forensics/   # ELA + metadata tamper detection (Monish)
├── forensics-log/        # Hash-chained chain-of-custody module (Sanskrut)
├── certificate-gen/      # Section 63(4) certificate auto-drafting (Sanskrut)
├── backend/               # FastAPI server tying everything together (Bhushan)
├── frontend/              # Upload + results dashboard (Bhushan)
├── docs/                  # Synopsis, diagrams, reports
└── README.md
```

## Tech stack

- **AI/ML:** PyTorch, OpenCV, facenet-pytorch (MTCNN), pytorch-grad-cam
- **Datasets:** FaceForensics++ (C23), Celeb-DF v2
- **Backend:** FastAPI
- **Database/logging:** SQLite + hashlib (SHA-256)
- **Certificate generation:** ReportLab / WeasyPrint
- **Frontend:** React (or plain HTML/CSS/JS)

## Setup

Each module has its own `requirements.txt`. To get started with the detection model pipeline:

```bash
cd detection-model/data_prep
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See `detection-model/data_prep/README.md` for dataset preparation instructions.

## Status

Project in progress — final-year BCA project, 5-month timeline.

## Legal grounding

This project is built around a documented gap in India's electronic evidence law: Section 63 of the BSA, 2023 authenticates that evidence hasn't been altered *since capture*, but does not test whether content was AI-generated *at* capture. See `/docs` for full literature survey and references.
