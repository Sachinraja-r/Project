import cv2, os, json
import numpy as np
from PIL import Image
import database

HAAR_FILE    = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
DATASETS     = os.path.join(os.path.dirname(__file__), 'datasets')
TRAINER_DIR  = os.path.join(os.path.dirname(__file__), 'trainer')
os.makedirs(TRAINER_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(HAAR_FILE)
recognizer   = cv2.face.LBPHFaceRecognizer_create()

def load_training_data(dataset_path):
    face_samples = []
    labels       = []
    label_map    = {}      # name → int id
    label_id     = 0

    for student_name in sorted(os.listdir(dataset_path)):
        student_dir = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_dir):
            continue

        label_map[student_name] = label_id
        database.register_student_db(student_name)
        image_count = 0

        for img_file in os.listdir(student_dir):
            if not img_file.lower().endswith('.png'):
                continue
            img_path = os.path.join(student_dir, img_file)
            try:
                img    = Image.open(img_path).convert('L')
                img_np = np.array(img, 'uint8')
                face_samples.append(img_np)
                labels.append(label_id)
                image_count += 1
            except Exception as e:
                print(f"  Skipped {img_file}: {e}")

        print(f"  [{label_id}] {student_name} — {image_count} face samples loaded")
        label_id += 1

    return face_samples, labels, label_map

print("=" * 50)
print("  ProctorEye Pro — Training Face Recognizer")
print("=" * 50)

if not os.path.isdir(DATASETS) or not os.listdir(DATASETS):
    raise SystemExit("No student data found. Run register.py first.")

print("\nLoading training data...")
faces, labels, label_map = load_training_data(DATASETS)

if not faces:
    raise SystemExit("No face samples found. Check datasets/ folder.")

print(f"\nTraining LBPH model on {len(faces)} samples...")
recognizer.train(faces, np.array(labels))

model_path = os.path.join(TRAINER_DIR, 'trainer.yml')
recognizer.write(model_path)
print(f"Model saved: {model_path}")

labels_path = os.path.join(TRAINER_DIR, 'labels.json')
with open(labels_path, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"Labels saved: {labels_path}")

print(f"\nTraining complete.")
print(f"  Students registered : {len(label_map)}")
print(f"  Total face samples  : {len(faces)}")
print(f"  Label map           : {label_map}")
print("\nNext step: Launch FastAPI Web Dashboard  python server.py")
