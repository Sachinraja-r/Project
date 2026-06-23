#CONVERTING NORMAL IMAGE TO GRAY IMAGE

import cv2

img=cv2.imread("google.png")

cv2.imshow("google",img)

cv2.waitKey(5000)
cv2.destroyAllWindows()
