import cv2
import numpy as np
import time, os, math
import handtrackingModule as htm

folderPath="header"
myList=os.listdir(folderPath)
# print(myList)
overLayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overLayList.append(image)
# print(len(overLayList))
header=overLayList[0]
DrawColor=(119, 6, 96)
brush = 15
erasorThickness = 100
xp,yp=0,0

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,800)
detector=htm.handDetector(detectionCon=0.85)
imgcanvas=np.zeros((720,1280,3),np.uint8)

while True:
    #1. Import the image
    success, img=cap.read()
    img= cv2.flip(img,1) 

    #2. Find hand landmarks using handtracking module
    img=detector.findHands(img)
    lmList=detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        # print(lmList)

        x1,y1= lmList[8][1],lmList[8][2]  #tip of the index finger
        x2,y2=lmList[12][1],lmList[12][2]  #tip of the middle finger

        #3. To find which fingers are up! (1: to draw and 2: to select the brush)
        fingers=detector.fingersUp()
        print(fingers)
        
        #4. if selection mode -> 2 fingers are up 
        if fingers[1] and fingers[2] and fingers[3]==False and fingers[4]==False and fingers[0]==False:
            xp,yp=0,0
            print("selection mode")
            #checking for the click
            if y1 < 125: 
                if 200<x1<400:
                    header=overLayList[0]
                    DrawColor=(119, 6, 96)
                elif 450<x1<600:
                    header=overLayList[1]
                    DrawColor=(196, 25, 8)
                elif 630 <x1< 770:
                    header=overLayList[2]
                    DrawColor=(13, 27, 93)
                elif 800<x1<930:
                    header=overLayList[3]
                    DrawColor=(6, 119, 31)
                elif 970< x1< 1250:
                    header=overLayList[4]
                    DrawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),DrawColor,cv2.FILLED)
                    
        #5. id drawing mode -> Index finger is up to draw/paint
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1-20),15,DrawColor,cv2.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1 
            if DrawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),DrawColor,erasorThickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),DrawColor,erasorThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),DrawColor,brush)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),DrawColor,brush)
            xp,yp=x1,y1  #previous points
    imgGray=cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)    
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgcanvas)

    #6. setting the header img (the nav bar)
    img[0:125,0:1280] = header
    img=cv2.addWeighted(img,0.5,imgcanvas,0.5,0) 
    cv2.imshow("img",img)
    cv2.imshow("canvas",imgcanvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break



