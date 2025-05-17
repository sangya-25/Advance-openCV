import cv2 
import mediapipe as mp
import time
import math

#initializing everything under this class which are requisite
class handDetector():
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )   #this object only uses RGB images
        self.mpDraw=mp.solutions.drawing_utils  #it is use to draw points over different landmarks
        self.tipIds=[4,8,12,16,20]


    #a function to detect hands and then to draw points for landmarks and connecting them only when asked!  because: (draw = True)
    def findHands(self, img, draw=True):  
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)  #it wil detect multiple hands in a frame
        if self.results.multi_hand_landmarks:
            for singleHand in self.results.multi_hand_landmarks:
                if draw:
                    landmark_spec = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=1) 
                    connection_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    self.mpDraw.draw_landmarks(img, singleHand, self.mpHands.HAND_CONNECTIONS,landmark_spec, connection_spec)
        return img    
                

    def findPosition(self, img, handNo=0, draw=True):
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            xList = []
            yList = []
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            if xList and yList:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = [xmin, ymin, xmax, ymax]

                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                    (0, 255, 0), 2)

        return self.lmList, bbox
    
    def fingersUp(self):
       
        fingers=[]
        if len(self.lmList) == 0:
            return fingers 
        #for thumb : not checking on the basis of y-parameter rather checking if point 4 goes beyond the point 3 then considering to be open
        #else considering top be closed(on the right of point 3 of thumb)
        if self.lmList[4][1] < self.lmList[4-1][1]:  
            fingers.append(0)
            # print("Index finger open")    
        else:
            fingers.append(1) 

        #for the remaining fingers:
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:  #since in openCV y distance is measured from the top of the window
                fingers.append(1)
                # print("Index finger open")    
            else:
                fingers.append(0) 
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    cTime=0
    detector=handDetector()
    while True:
        success, img=cap.read()
        img=detector.findHands(img)
        lmList,bbox=detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0),3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__=="__main__":
    main()