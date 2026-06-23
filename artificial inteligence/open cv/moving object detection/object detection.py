import cv2
import imutils
import time

vc=cv2.VideoCapture(0)
time.sleep(1)

firstframe=None
area = 500

while True:
     _,cp=vc.read()
     text="Normal"
     cp=imutils.resize(cp,width=1000)
     gray=cv2.cvtColor(cp,cv2.COLOR_BGR2GRAY)
     gaus=cv2.GaussianBlur(gray,(21,21),0)
     if firstframe is None:
         firstframe=gaus
         continue
     imgdiff=cv2.absdiff(firstframe,gaus)
     thresh=cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
     thresh=cv2.dilate(thresh,None,iterations=2)
     cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
     cnts=imutils.grab_contours(cnts)
     for c in cnts:
        if cv2.contourArea(c)<area :
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(cp,(x,y),(x+w,y+h),(0,255,0),2)
        text="moving object"
     print(text)
     cv2.putText(cp,text,(10,20),
     cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
     cv2.imshow("camera",cp)
     key=cv2.waitKey(1)
     if key == ord("l"):
        break
cp.release()
cv.destroyAllWindows()
