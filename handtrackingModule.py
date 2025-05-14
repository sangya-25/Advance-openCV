import cv2 
import mediapipe as mp
import time


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
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo] #it will detect the first hand 
            for id, lm in enumerate(myHand.landmark): #it will return the id for each and evry landmark in a hand
                    #print(id,lm) 
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w), int(lm.y*h)  # to get the value in integer instead of floating values
                    #multiplied x value of landmark with width and y with height to get the exact central location 
                    lmList.append([id, cx,cy])
                    #to highlight a particular landmark
                    if draw: 
                        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)

        return lmList


def main():
    cap=cv2.VideoCapture(0)
    pTime=0
    cTime=0
    detector=handDetector()
    while True:
        success, img=cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
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