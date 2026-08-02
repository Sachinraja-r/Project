import cv2
import os
import json
import time
import math
import datetime
import threading
import numpy as np
import imutils

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from facial_emotion_recognition import EmotionRecognition
    HAS_FER = True
except Exception:
    HAS_FER = False


HAAR_FILE    = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
TRAINER_PATH = os.path.join(os.path.dirname(__file__), 'trainer', 'trainer.yml')
LABELS_PATH  = os.path.join(os.path.dirname(__file__), 'trainer', 'labels.json')
ALERT_DIR    = os.path.join(os.path.dirname(__file__), 'alerts')

os.makedirs(ALERT_DIR, exist_ok=True)

class ProctorEngine:
    def __init__(self, student_name: str, session_id: str):
        self.student_name = student_name
        self.session_id   = session_id
        
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.id_to_name = {}
        self.expected_id = None
        
        if os.path.exists(TRAINER_PATH) and os.path.exists(LABELS_PATH):
            self.recognizer.read(TRAINER_PATH)
            with open(LABELS_PATH, 'r') as f:
                label_map = json.load(f)
                self.id_to_name = {v: k for k, v in label_map.items()}
                if student_name in label_map:
                    self.expected_id = label_map[student_name]

        self.mp_face_mesh = None
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=2,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"MediaPipe initialization warning: {e}")
                self.mp_face_mesh = None

        self.yolo_model = None
        if HAS_YOLO:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"YOLO initialization warning: {e}")
                self.yolo_model = None

        self.er_model = None
        if HAS_FER:
            try:
                self.er_model = EmotionRecognition(device='cpu')
            except Exception as e:
                print(f"FER initialization warning: {e}")
                self.er_model = None

        self.frame_num = 0
        self.start_time = datetime.datetime.now()
        self.absence_frames = 0
        self.ABSENCE_LIMIT = 90
        
        self.cached_name = "Detecting..."
        self.cached_conf = 0.0
        self.cached_emotion = "neutral"
        self.cached_yolo_objects = []
        self.cached_gaze_zone = "CENTER"
        
        self.latest_frame_for_worker = None
        self.worker_running = True
        
        self.alert_counts = {
            "wrong_face": 0,
            "no_face": 0,
            "gaze": 0,
            "emotion": 0,
            "phone": 0,
            "multi_person": 0,
            "audio": 0
        }
        
        self.penalty_points = 0.0
        self.integrity_score = 100.0
        self.attention_score = 100.0

        self.worker_thread = threading.Thread(target=self._async_heavy_models_loop, daemon=True)
        self.worker_thread.start()

    def _async_heavy_models_loop(self):
        while self.worker_running:
            if self.latest_frame_for_worker is not None:
                frame = self.latest_frame_for_worker.copy()
                
                if self.yolo_model:
                    try:
                        results = self.yolo_model.predict(frame, verbose=False, conf=0.45, imgsz=320)
                        detected = []
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                cls_name = self.yolo_model.names[cls_id]
                                if cls_name in ["cell phone", "book"]:
                                    detected.append(cls_name)
                        self.cached_yolo_objects = list(set(detected))
                    except Exception:
                        pass
                
                if self.er_model:
                    try:
                        emo_dict = self.er_model.recognise_emotion(frame, return_type='dictionary')
                        if emo_dict and isinstance(emo_dict, dict):
                            self.cached_emotion = max(emo_dict, key=emo_dict.get)
                    except Exception:
                        pass
                        
            time.sleep(0.2)

    def stop(self):
        self.worker_running = False

    def calculate_iris_gaze(self, frame_rgb, h, w):
        if not self.mp_face_mesh:
            return None, 1
            
        results = self.mp_face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None, 0
            
        face_count = len(results.multi_face_landmarks)
        mesh_points = results.multi_face_landmarks[0].landmark

        left_iris = mesh_points[468]
        left_inner = mesh_points[133]
        left_outer = mesh_points[33]
        
        total_eye_w = abs(left_inner.x - left_outer.x)
        if total_eye_w > 0:
            iris_pos = (left_iris.x - left_outer.x) / total_eye_w
        else:
            iris_pos = 0.5
            
        gaze = "CENTER"
        if iris_pos < 0.35:
            gaze = "LOOKING RIGHT"
        elif iris_pos > 0.65:
            gaze = "LOOKING LEFT"
            
        return gaze, face_count

    def process_frame(self, raw_frame):
        self.frame_num += 1
        frame = imutils.resize(raw_frame, width=600)
        h, w = frame.shape[:2]
        
        self.latest_frame_for_worker = frame
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        alert_level = "CLEAR"
        alert_reason = ""
        gaze_zone = "CENTER"

        mp_gaze, mp_face_count = self.calculate_iris_gaze(frame_rgb, h, w)
        if mp_gaze:
            gaze_zone = mp_gaze
            
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 4)
        num_faces = max(len(faces), mp_face_count)

        if num_faces == 0:
            self.absence_frames += 1
            self.cached_name = "No face"
            gaze_zone = "OUT OF FRAME"
            
            if self.absence_frames >= self.ABSENCE_LIMIT:
                alert_level = "HIGH"
                alert_reason = "no_face"
                if self.absence_frames == self.ABSENCE_LIMIT:
                    self.alert_counts["no_face"] += 1
                    self.penalty_points += 3.0
                    self.save_snapshot(frame, "no_face")
        else:
            self.absence_frames = 0

            if num_faces > 1:
                alert_level = "HIGH"
                alert_reason = "multi_person"
                self.alert_counts["multi_person"] += 1
                self.penalty_points += 3.0
                self.save_snapshot(frame, "multi_person")

            for (x, y, fw, fh) in faces:
                if self.frame_num % 5 == 0 and self.expected_id is not None:
                    face_gray = cv2.resize(gray[y:y+fh, x:x+fw], (130, 100))
                    label, conf = self.recognizer.predict(face_gray)
                    self.cached_conf = round(conf, 1)
                    if conf < 70 and label == self.expected_id:
                        self.cached_name = self.student_name
                    else:
                        self.cached_name = self.id_to_name.get(label, "Unknown") if conf < 70 else "Unknown"

                if self.cached_name != self.student_name and self.cached_name not in ("Detecting...", "No face"):
                    alert_level = "HIGH"
                    alert_reason = "wrong_face"
                    self.alert_counts["wrong_face"] += 1
                    self.penalty_points += 3.0
                    self.save_snapshot(frame, "wrong_face")
                    box_clr = (30, 30, 220)
                else:
                    box_clr = (0, 220, 80)

                if not mp_gaze:
                    face_cx = x + fw // 2
                    if face_cx < w * 0.28:
                        gaze_zone = "LOOKING RIGHT"
                    elif face_cx > w * 0.72:
                        gaze_zone = "LOOKING LEFT"

                if gaze_zone != "CENTER" and alert_level == "CLEAR":
                    alert_level = "MEDIUM"
                    alert_reason = "gaze"
                    if self.frame_num % 15 == 0:
                        self.alert_counts["gaze"] += 1
                        self.penalty_points += 1.5

                cv2.rectangle(frame, (x, y), (x+fw, y+fh), box_clr, 2)
                cv2.putText(frame, f"{self.cached_name} ({self.cached_conf})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_clr, 1)

        detected_objects = self.cached_yolo_objects

        if "cell phone" in detected_objects:
            alert_level = "HIGH"
            alert_reason = "phone"
            if self.frame_num % 30 == 0:
                self.alert_counts["phone"] += 1
                self.penalty_points += 3.0
                self.save_snapshot(frame, "phone")

        if self.cached_emotion in {'angry', 'fear', 'disgust', 'surprise'} and alert_level in ('CLEAR', 'LOW'):
            alert_level = "LOW"
            alert_reason = "emotion"
            if self.frame_num % 30 == 0:
                self.alert_counts["emotion"] += 1
                self.penalty_points += 0.5

        self.integrity_score = max(0.0, round(100.0 - (self.penalty_points / max(1.0, self.frame_num * 0.05)) * 100, 1))
        if alert_level == "CLEAR":
            self.attention_score = min(100.0, round(self.attention_score + 0.2, 1))
        elif alert_level == "LOW":
            self.attention_score = max(0.0, round(self.attention_score - 0.5, 1))
        elif alert_level == "MEDIUM":
            self.attention_score = max(0.0, round(self.attention_score - 1.5, 1))
        elif alert_level == "HIGH":
            self.attention_score = max(0.0, round(self.attention_score - 3.5, 1))

        elapsed = str(datetime.datetime.now() - self.start_time).split('.')[0]
        frame = self.draw_hud(frame, self.cached_name, self.cached_emotion, gaze_zone,
                              alert_level, elapsed, detected_objects)

        telemetry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "frame": self.frame_num,
            "student": self.cached_name,
            "confidence": self.cached_conf,
            "emotion": self.cached_emotion,
            "gaze": gaze_zone,
            "alert_level": alert_level,
            "alert_reason": alert_reason,
            "integrity_score": self.integrity_score,
            "attention_score": self.attention_score,
            "objects_detected": detected_objects,
            "counts": self.alert_counts
        }

        return frame, telemetry

    def alert_color(self, level):
        return {
            'CLEAR' : (0, 200, 80),
            'LOW'   : (0, 200, 255),
            'MEDIUM': (0, 130, 255),
            'HIGH'  : (30, 30, 220)
        }.get(level, (0, 200, 80))

    def draw_hud(self, frame, name, emotion, gaze, alert_lvl, elapsed, objects):
        h, w = frame.shape[:2]
        al_clr = self.alert_color(alert_lvl)

        cv2.rectangle(frame, (0, 0), (w, 95), (15, 15, 15), -1)
        cv2.putText(frame, f"Student : {self.student_name} ({name})", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Emotion : {emotion}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"Gaze    : {gaze}", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.putText(frame, f"Time: {elapsed}", (w-190, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Score: {self.integrity_score}%", (w-190, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

        bx = w - 190
        cv2.rectangle(frame, (bx, 58), (w-12, 88), al_clr, -1)
        cv2.rectangle(frame, (bx, 58), (w-12, 88), (0,0,0), 1)
        cv2.putText(frame, f"ALERT: {alert_lvl}", (bx+8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,0,0), 2)

        cv2.rectangle(frame, (0, h-32), (w, h), (15,15,15), -1)
        summary = (f"Wrong face: {self.alert_counts['wrong_face']}  "
                   f"No face: {self.alert_counts['no_face']}  "
                   f"Gaze: {self.alert_counts['gaze']}  "
                   f"Phone: {self.alert_counts['phone']}")
        cv2.putText(frame, summary, (12, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        if objects:
            cv2.putText(frame, f"⚠️ DETECTED: {', '.join(objects).upper()}", (w-260, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

        return frame

    def save_snapshot(self, frame, reason):
        ts = datetime.datetime.now().strftime('%H%M%S')
        path = os.path.join(ALERT_DIR, f"{self.session_id}_{ts}_{reason}.png")
        cv2.imwrite(path, frame)
        return path
