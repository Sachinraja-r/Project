import cv2
img=cv2.imread("google.png")
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("original",img)
cv2.imshow("grayimgcvt",grayimg)
cv2.imwrite("grayimage.png",grayimg)
