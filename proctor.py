# ─────────────────────────────────────────────────────────────
#  ProctorEye | proctor.py
#  Step 3 — Run the live proctoring session
#
#  Modules combined:
#    Face Recognition  → continuous identity verification
#    Emotion Detection → suspicious behavior flagging
#    Gaze Tracking     → spatial zone logic (from object tracker)
#
#  Press ESC to end session and save log
# ─────────────────────────────────────────────────────────────

import cv2, os, csv, json, datetime
import numpy as np
import imutils
from facial_emotion_recognition import EmotionRecognition

# ── CONFIG ────────────────────────────────────────────────────
HAAR_FILE            = 'haarcascade_frontalface_default.xml'
TRAINER_PATH         = 'trainer/trainer.yml'
LABELS_PATH          = 'trainer/labels.json'
LOG_DIR              = 'logs'
ALERT_DIR            = 'alerts'

CONFIDENCE_THRESHOLD = 70     # LBPH: lower value = higher confidence match
GAZE_MARGIN          = 0.28   # fraction of frame width defining center zone
ABSENCE_LIMIT        = 90     # frames before "no face" alert (~3s at 30fps)
CHECK_EVERY_N_FRAMES = 5      # run face recognition every N frames (performance)

# Emotions that flag suspicious behavior
SUSPICIOUS_EMOTIONS  = {'angry', 'disgust', 'fear', 'surprise'}

# Emotion label map from facial_emotion_recognition indices
FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# ─────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ALERT_DIR, exist_ok=True)


# ── Load models ───────────────────────────────────────────────
print("Loading models...")
face_cascade = cv2.CascadeClassifier(HAAR_FILE)

recognizer   = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)

# facial_emotion_recognition handles both detection & display overlay
er = EmotionRecognition(device='cpu')

with open(LABELS_PATH) as f:
    label_map = json.load(f)                         # { name: int_id }
id_to_name = {v: k for k, v in label_map.items()}   # { int_id: name }

print(f"Registered students: {list(label_map.keys())}")


# ── Session setup ─────────────────────────────────────────────
student_name = input("\nEnter student name for this session: ").strip()
if student_name not in label_map:
    raise SystemExit(f"'{student_name}' not registered. Run register.py first.")

expected_id   = label_map[student_name]
session_time  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path      = os.path.join(LOG_DIR, f'session_{session_time}.csv')

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    raise SystemExit("Cannot access webcam.")

print(f"\nProctorEye started for: {student_name}")
print("Press ESC to end the exam.\n")


# ── State variables ───────────────────────────────────────────
log_rows        = []
alert_counts    = {'wrong_face': 0, 'no_face': 0, 'gaze': 0, 'emotion': 0}
absence_frames  = 0
frame_num       = 0
start_time      = datetime.datetime.now()

# Cached values (updated every CHECK_EVERY_N_FRAMES)
cached_name     = 'Detecting...'
cached_conf     = 0.0
cached_emotion  = 'neutral'


# ── Helper functions ──────────────────────────────────────────
def get_gaze_zone(face_cx, frame_width):
    """
    Reuses directional logic from object tracker:
    Divides frame into LEFT | CENTER | RIGHT zones.
    Face center in left zone → student looking away (left side = right on mirrored cam).
    """
    left_bound  = frame_width * GAZE_MARGIN
    right_bound = frame_width * (1 - GAZE_MARGIN)
    if face_cx < left_bound:
        return 'LOOKING RIGHT'
    elif face_cx > right_bound:
        return 'LOOKING LEFT'
    return 'CENTER'

def alert_color(level):
    return {
        'CLEAR' : (0, 200, 80),
        'LOW'   : (0, 200, 255),
        'MEDIUM': (0, 130, 255),
        'HIGH'  : (30, 30, 220)
    }.get(level, (0, 200, 80))

def save_snapshot(frame, reason):
    ts   = datetime.datetime.now().strftime('%H%M%S')
    path = os.path.join(ALERT_DIR, f'{ts}_{reason}.png')
    cv2.imwrite(path, frame)

def draw_hud(frame, name, emotion, gaze, alert_lvl, elapsed, counts):
    h, w = frame.shape[:2]
    al_clr = alert_color(alert_lvl)

    # ── Top status bar ──
    cv2.rectangle(frame, (0, 0), (w, 95), (15, 15, 15), -1)

    cv2.putText(frame, f"Student : {name}",  (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1)
    cv2.putText(frame, f"Emotion : {emotion}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180,180,180), 1)
    cv2.putText(frame, f"Gaze    : {gaze}", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180,180,180), 1)

    cv2.putText(frame, elapsed, (w-140, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1)

    # Alert badge
    bx = w - 145
    cv2.rectangle(frame, (bx, 38), (w-12, 84), al_clr, -1)
    cv2.rectangle(frame, (bx, 38), (w-12, 84), (0,0,0), 1)
    cv2.putText(frame, alert_lvl, (bx+8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

    # ── Bottom alert counter bar ──
    cv2.rectangle(frame, (0, h-28), (w, h), (15,15,15), -1)
    summary = (f"Wrong face: {counts['wrong_face']}   "
               f"No face: {counts['no_face']}   "
               f"Gaze: {counts['gaze']}   "
               f"Emotion: {counts['emotion']}")
    cv2.putText(frame, summary, (12, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

    return frame


# ── Main loop ─────────────────────────────────────────────────
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame   = imutils.resize(frame, width=820)
    fh, fw  = frame.shape[:2]
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces   = face_cascade.detectMultiScale(gray, 1.3, 4)

    alert_level  = 'CLEAR'
    alert_reason = ''
    gaze_zone    = 'N/A'

    # ── Emotion detection (every N frames) ──
    if frame_num % CHECK_EVERY_N_FRAMES == 0:
        try:
            emo_result = er.recognise_emotion(frame, return_type='dictionary')
            if emo_result and isinstance(emo_result, dict):
                # Returns {label: score, ...} — pick highest
                cached_emotion = max(emo_result, key=emo_result.get)
            elif emo_result and isinstance(emo_result, (int, float)):
                idx = int(emo_result)
                cached_emotion = FER_LABELS[idx] if 0 <= idx < len(FER_LABELS) else 'neutral'
        except Exception:
            cached_emotion = 'neutral'

    # ── No face detected ──
    if len(faces) == 0:
        absence_frames += 1
        cached_name     = 'No face'
        gaze_zone       = 'OUT OF FRAME'
        if absence_frames >= ABSENCE_LIMIT:
            alert_level  = 'HIGH'
            alert_reason = 'no_face'
            if absence_frames == ABSENCE_LIMIT:             # save once per event
                alert_counts['no_face'] += 1
                save_snapshot(frame, 'no_face')
    else:
        absence_frames = 0

        for (x, y, w, h) in faces:
            # ── Face recognition (every N frames) ──
            if frame_num % CHECK_EVERY_N_FRAMES == 0:
                face_gray  = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                label, conf = recognizer.predict(face_gray)
                cached_conf = round(conf, 1)
                if conf < CONFIDENCE_THRESHOLD and label == expected_id:
                    cached_name = student_name
                else:
                    cached_name = id_to_name.get(label, 'Unknown') if conf < CONFIDENCE_THRESHOLD else 'Unknown'

            # Alert on wrong face
            if cached_name != student_name and cached_name not in ('Detecting...', 'No face'):
                alert_level  = 'HIGH'
                alert_reason = 'wrong_face'
                alert_counts['wrong_face'] += 1
                save_snapshot(frame, 'wrong_face')
                box_color = (30, 30, 220)
            else:
                box_color = (0, 220, 80)

            # ── Gaze zone (reused from object tracker directional logic) ──
            face_cx   = x + w // 2
            gaze_zone = get_gaze_zone(face_cx, fw)
            if gaze_zone != 'CENTER' and alert_level == 'CLEAR':
                alert_level  = 'MEDIUM'
                alert_reason = 'gaze'
                alert_counts['gaze'] += 1

            # Draw face bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, f"{cached_name} ({cached_conf})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, box_color, 1)

            # Gaze zone line markers on frame
            cx_left  = int(fw * GAZE_MARGIN)
            cx_right = int(fw * (1 - GAZE_MARGIN))
            cv2.line(frame, (cx_left, 95), (cx_left, fh-28), (80,80,80), 1)
            cv2.line(frame, (cx_right, 95), (cx_right, fh-28), (80,80,80), 1)

        # ── Emotion alert (if no higher alert already set) ──
        if cached_emotion in SUSPICIOUS_EMOTIONS and alert_level in ('CLEAR', 'LOW'):
            alert_level  = 'LOW'
            alert_reason = 'emotion'
            alert_counts['emotion'] += 1

    # ── Draw HUD overlay ──
    elapsed = str(datetime.datetime.now() - start_time).split('.')[0]
    frame   = draw_hud(frame, cached_name, cached_emotion,
                       gaze_zone, alert_level, elapsed, alert_counts)

    # ── Log this frame ──
    log_rows.append([
        datetime.datetime.now().strftime('%H:%M:%S'),
        frame_num,
        cached_name,
        cached_conf,
        cached_emotion,
        gaze_zone,
        alert_level,
        alert_reason
    ])

    cv2.imshow('ProctorEye — Live Proctoring', frame)
    frame_num += 1

    if cv2.waitKey(1) == 27:
        print("\nSession ended by user.")
        break


# ── Cleanup & save log ────────────────────────────────────────
webcam.release()
cv2.destroyAllWindows()

with open(log_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time','frame','face_detected','confidence',
                     'emotion','gaze','alert_level','reason'])
    writer.writerows(log_rows)

duration = str(datetime.datetime.now() - start_time).split('.')[0]
print(f"\n{'='*50}")
print(f"  Session Summary")
print(f"{'='*50}")
print(f"  Student   : {student_name}")
print(f"  Duration  : {duration}")
print(f"  Frames    : {frame_num}")
print(f"  Log saved : {log_path}")
print(f"\n  Alert counts:")
for k, v in alert_counts.items():
    print(f"    {k:<15}: {v}")
print(f"\nNext step: Run  python report.py")
