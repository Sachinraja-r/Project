import cv2
import imutils
img=cv2.imread("google.png")
resize=imutils.resize(img,width=200)
cv2.imshow("orgiginal",img)
cv2.imshow("resize",resize)
cv2.imwrite("resize img.png",resize)
