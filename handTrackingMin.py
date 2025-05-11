import cv2 
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()   #this object only uses RGB images
mpDraw=mp.solutions.drawing_utils  #it is use to draw points over different landmarks

pTime=0
cTime=0

while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)  #it wil detect multiple hands in a frame
    if results.multi_hand_landmarks:
        for singleHand in results.multi_hand_landmarks:
            for id, lm in enumerate(singleHand.landmark): #it will return the id for each and evry landmark in a hand
                #print(id,lm) 
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)  # to get the value in integer instead of floating values
                #multiplied x value of landmark with width and y with height to get the exact central location 

                #to highlight a particular landmark
                if(id==0): 
                    cv2.circle(img,(cx,cy),30,(205,0,205),cv2.FILLED)
            mpDraw.draw_landmarks(img, singleHand, mpHands.HAND_CONNECTIONS)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)