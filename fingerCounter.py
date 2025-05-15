import cv2
import time , math, os
import handtrackingModule as htm

wCam, hCam=640,480
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
detector=htm.handDetector(detectionCon=0.75)
tipIds=[4,8,12,16,20]  #tips of all the finger



folderPath="FingerImages"
myList=os.listdir(folderPath)
print(myList)
overLayList=[]
for imgPath in myList:
    image=cv2.imread(f'{folderPath}/{imgPath}')
    image = cv2.resize(image, (200, 200))
    overLayList.append(image)
print(len(overLayList))

while True:
    success, img=cap.read()
    img=detector.findHands(img)
    lmlist=detector.findPosition(img,draw=False)

    #taking landmarks of all the finger tips i.e 4,8,12,16 and 20 
    #and checking one by one that if the tip is below pip landmark of respective fingers or not
    #if yes that it is considered to be closed else open

    if len(lmlist)!=0:
        fingers=[]
        #for thumb : not checking on the basis of y-parameter rather checking if point 4 goes beyond the point 3 then considering to be open
        #else considering top be closed(on the right of point 3 of thumb)
        if lmlist[4][1] < lmlist[4-1][1]:  
            fingers.append(0)
            # print("Index finger open")    
        else:
            fingers.append(1) 

        #for the remaining fingers:
        for id in range(1,5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:  #since in openCV y distance is measured from the top of the window
                fingers.append(1)
                # print("Index finger open")    
            else:
                fingers.append(0) 
        totalFindersUp=fingers.count(1)
        print(totalFindersUp)
        h,w,c=overLayList[totalFindersUp-1].shape
        img[0:h, 0:w] = overLayList[totalFindersUp-1]
        cv2.rectangle(img,(5,200),(150,300),(0,255,0),cv2.FILLED)
        if totalFindersUp==0:
            cv2.putText(img,f'Zero',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        elif totalFindersUp==1:
            cv2.putText(img,f'One',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        elif totalFindersUp==2:
            cv2.putText(img,f'Two',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        elif totalFindersUp==3:
            cv2.putText(img,f'Three',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        elif totalFindersUp==4:
            cv2.putText(img,f'Four',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(img,f'Five',(40,260),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(500,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow("img",img)
    cv2.waitKey(1)