import cv2, os
import database

HAAR_FILE = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
DATASETS  = os.path.join(os.path.dirname(__file__), 'datasets')
WIDTH, HEIGHT = 130, 100
TOTAL_IMAGES  = 50

student_name = input("Enter student name: ").strip()
if not student_name:
    raise SystemExit("Name cannot be empty.")

save_path = os.path.join(DATASETS, student_name)
os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(HAAR_FILE)
webcam       = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise SystemExit("Cannot access webcam.")

count = 1
print(f"\nCapturing {TOTAL_IMAGES} images for '{student_name}'.")
print("Look directly at the camera. Press ESC to cancel.\n")

while count <= TOTAL_IMAGES:
    ret, frame = webcam.read()
    if not ret:
        continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        face        = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (WIDTH, HEIGHT))
        cv2.imwrite(os.path.join(save_path, f"{count}.png"), face_resize)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1

    # Progress bar
    progress = int((min(count-1, TOTAL_IMAGES) / TOTAL_IMAGES) * 300)
    cv2.rectangle(frame, (10, frame.shape[0]-30), (310, frame.shape[0]-10), (50,50,50), -1)
    cv2.rectangle(frame, (10, frame.shape[0]-30), (10+progress, frame.shape[0]-10), (0,200,100), -1)

    cv2.putText(frame, f"Capturing: {min(count-1,TOTAL_IMAGES)}/{TOTAL_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Student: {student_name}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 255, 255), 2)

    cv2.imshow("ProctorEye - Register Student", frame)
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

saved = count - 1
print(f"Done! {saved} images saved to: {save_path}")
if saved > 0:
    database.register_student_db(student_name)
    print(f"Registered '{student_name}' in ProctorEye database.")
    print("Next step: Run  python train.py")
else:
    print("Warning: No images captured.")
