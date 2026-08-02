# ProctorEye 👁️
### AI-Powered Online Exam Proctoring System

A real-time computer vision proctoring system that combines **face recognition**,
**emotion detection**, and **gaze tracking** to monitor exam integrity — and generates
a PDF report at the end of every session.

---

## How It Works

```
Webcam Feed
    │
    ├── Face Recognition   → Is this the registered student? (LBPH + Haar)
    ├── Emotion Detection  → Any suspicious emotion? (FER)
    └── Gaze Tracking      → Is the student looking away? (Spatial zone logic)
                │
                ▼
        Alert Engine  →  CLEAR / LOW / MEDIUM / HIGH
                │
                ▼
      CSV Event Log  +  Alert Snapshots  →  PDF Report
```

---

## Alerts

| Alert Level | Trigger                                     |
|-------------|---------------------------------------------|
| 🔴 HIGH     | Wrong face / Face absent > 3 seconds        |
| 🟠 MEDIUM   | Gaze deviation (looking left / right / up)  |
| 🟡 LOW      | Suspicious emotion (angry, fear, disgust)   |
| 🟢 CLEAR    | All checks passed                           |

---

## Project Structure

```
ProctorEye/
├── register.py       ← Step 1: Capture student face images
├── train.py          ← Step 2: Train LBPH face recognizer
├── proctor.py        ← Step 3: Run live proctoring session
├── report.py         ← Step 4: Generate PDF integrity report
├── requirements.txt
├── haarcascade_frontalface_default.xml
├── datasets/         ← Auto-created: student face images
├── trainer/          ← Auto-created: trained model + labels
├── logs/             ← Auto-created: session CSV logs
├── alerts/           ← Auto-created: alert snapshots
└── reports/          ← Auto-created: PDF reports
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Haar cascade (if not present)
# Place haarcascade_frontalface_default.xml in the project root
# Download from: https://github.com/opencv/opencv/tree/master/data/haarcascades
```

---

## Usage — 4 Steps

### Step 1 — Register student
```bash
python register.py
# Enter name when prompted → captures 50 face images
# Repeat for each student
```

### Step 2 — Train model
```bash
python train.py
# Reads all images from datasets/
# Saves trainer/trainer.yml and trainer/labels.json
```

### Step 3 — Run exam session
```bash
python proctor.py
# Select student name
# Live window shows: identity, emotion, gaze zone, alert level
# Press ESC to end → saves CSV log to logs/
```

### Step 4 — Generate report
```bash
python report.py
# Reads latest log from logs/
# Saves PDF to reports/ with integrity score + charts
```

---

## Resume Bullets

```
• Built ProctorEye, an AI proctoring system combining LBPH face recognition,
  FER-based emotion analysis, and spatial gaze tracking using OpenCV and Python.

• Engineered a 4-level alert pipeline (CLEAR/LOW/MEDIUM/HIGH) that detects
  identity mismatch, face absence, gaze deviation, and suspicious emotion in
  real time with sub-200ms frame latency.

• Automated session reporting with ReportLab — generates PDF integrity reports
  with per-student integrity scores, emotion distribution charts, and
  timestamped alert logs.
```

---

## Tech Stack

| Module              | Library                          |
|---------------------|----------------------------------|
| Face detection      | OpenCV Haar Cascade              |
| Face recognition    | OpenCV LBPH Recognizer           |
| Emotion detection   | FER + facial_emotion_recognition |
| Gaze tracking       | Spatial zone logic (custom)      |
| Event logging       | Python csv + datetime            |
| PDF report          | ReportLab                        |
| Frame processing    | imutils + NumPy                  |

---

## Integrity Score Formula

```
penalty = (HIGH × 3) + (MEDIUM × 1.5) + (LOW × 0.5)
score   = 100 − (penalty / max_penalty × 100)

≥ 75  →  PASS
< 75  →  REVIEW REQUIRED
```

---

*ProctorEye — For academic integrity use only.
AI-generated results should be reviewed by a human invigilator.*
