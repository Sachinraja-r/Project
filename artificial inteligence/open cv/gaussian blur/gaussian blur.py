import cv2

img=cv2.imread("google.png")

gaus1=cv2.GaussianBlur(img,(21,21),20)
gaus2=cv2.GaussianBlur(img,(41,41),0)
cv2.imshow("origiunal",img)
cv2.imshow("gaus1",gaus1)
cv2.imshow("gaus2",gaus2)
cv2.imwrite("gaus1.png",gaus1)
cv2.imwrite("gaus2.png",gaus2)
