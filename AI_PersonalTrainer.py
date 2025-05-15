#it is a AI based trainer using pose estimation to give the bicep exercise estimation(accuracy) based on the angle of the elbow and different arm landmarks
#Also, it calculates the bicep counting on each elbow movement detection
#basically counting the number of curls
import cv2
import time
import poseEstimationModule as ptm
import numpy as np

cap=cv2.VideoCapture('videos/p2.mp4')
if not cap.isOpened():
    print("❌ Error: Cannot open video file. Check the path or file format.")
    exit()
pTime=0
detector=ptm.poseDetector()
count=0
dir=0

while True:
    success , img=cap.read()
    if not success or img is None:
        print("✅ Video ended or failed to read frame.")
        break  # End of video or error
    img=cv2.resize(img,(800,800))
    # img=cv2.imread('AITrainer/test1.jpg')  
    img=detector.findPose(img,draw=False)
    lmList=detector.getPosition(img,draw=False)
    if len(lmList)!=0:
        # angle=detector.findAngle(img,11,13,15)  #for the left arm
        angle=detector.findAngle(img,12,14,16)   #for the right arm
        per=np.interp(angle,(68,140),(100,0))
        bar=np.interp(angle,(68,140),(100,700))
        # print(per,angle)

        #check for the dumbbell curls
        if per==100:
            if dir==0: #up direction
                count+=0.5
                dir=1
        if per==0:
            if dir==1:  #down direction
                count+=0.5
                dir=0
        print(count)
        cv2.rectangle(img,(700,100),(760,700),(255,0,255),2)
        cv2.rectangle(img,(700,int(bar)),(760,700),(255,0,255),cv2.FILLED)
        cv2.putText(img,f'{int(per)}%',(700,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.rectangle(img,(0,150),(340,50),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'Count: {int(count)}',(30,120),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow('image',img)
    cv2.waitKey(1)