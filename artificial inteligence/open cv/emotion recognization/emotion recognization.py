from facial_emotion_recognition import EmotionRecognition
import cv2

er=EmotionRecognition(device='cpu')
cam=cv2.VideoCapture(0)

while True:
    _,cm=cam.read()
    frame=er.recognise_emotion(cm,return_type='BGR')
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
