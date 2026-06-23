import cv2

img=cv2.imread("google.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh=cv2.threshold(gray,170,255,cv2.THRESH_BINARY)[1]
cv2.imshow("og",img)
cv2.imshow("gray",gray)
cv2.imshow("thresh img",thresh)
cv2.imwrite("gray.png",gray)
cv2.imwrite("thresholdimg.png",thresh)
