import cv2 
import poseEstimationModule as pm
import time

cap=cv2.VideoCapture('videos/4.mp4')
pTime=0
cTime=0
detector=pm.poseDetector()
while True:
    success, img=cap.read()
    img=detector.findPose(img)
    lmList=detector.getPosition(img)
    if len(lmList)!=0:
        cv2.circle(img,(lmList[14][1],lmList[14][2]),10,(0,0,255),cv2.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.imshow("image",img)
    cv2.waitKey(1)