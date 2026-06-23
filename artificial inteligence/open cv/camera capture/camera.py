import cv2

vc=cv2.VideoCapture(0)

while True:
    _,cp=vc.read()
    cv2.imshow("Live",cp)
    key=cv2.waitKey(1)
    print(key)
    if key == ord("l"):
        break
vc.release()
cv2.destroyAllWindows()
