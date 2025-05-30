import cv2
import numpy as np
import time, os, math
import autopy
import handtrackingModule as htm

################ Variables:
wCam, hCam = 640, 480
pTime=0
frameR = 100
smoothening = 5
plocX, plocY=0,0  #previous location
clocX, clocY=0,0  #current location
################

cap=cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)
detector=htm.handDetector(maxHands=1)
wScr, hScr= autopy.screen.size()


while True:
    success,img=cap.read()
    #1. Find the hand landmarks
    img=detector.findHands(img)
    lmList, bbox =detector.findPosition(img)
    
    #2. To get the tip of the index and the middle finger 
    if len(lmList)!=0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        #3. To check which fingers are up
        fingers=detector.fingersUp()
        print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        #4. Only Index Finger : In Moving Mode
        if fingers[1]==1 and fingers[2]==0:
            #5. Convert the coordinates
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            #6. Smoothen Values
            clocX=plocX+(x3-plocX)/smoothening
            clocY=plocY+(y3-plocY)/smoothening


            #7. Move Mouse 
            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX, plocY=clocX, clocY

        #8. Both index and middle fingers are up: Clicking Mode
        if fingers[1]==1 and fingers[2]==1:
            #9. Find distance btw the fingers
            length,img, lineInfo =detector.findDistance(8,12,img)
            print(length)
            #10. Click mouse if distance is short
            if length<35:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()


    #11. Frame rate
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    #12. Display
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break