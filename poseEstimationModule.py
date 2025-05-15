import cv2
import mediapipe as mp
import time
import math
# 11592987760
class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpdraw = mp.solutions.drawing_utils


    def findPose(self, img, draw=True):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if(self.results.pose_landmarks):
            if draw:
                landmark_spec = self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4) 
                connection_spec = self.mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                self.mpdraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, landmark_spec, connection_spec)
        return img
    

    def getPosition(self, img, draw=True):
        self.lmList=[]
        if self.results.pose_landmarks:
            for id , lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)  #to get the acutal pixel values in integer and not in ratio
                self.lmList.append([id,cx,cy])
                if draw:
                    #to ensure we are getting the right location for each landmarks, drawing circle over the positions to verify
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList
    
    def findAngle(self,img, p1,p2,p3, draw=True):
        #Get the requisite landmarks 
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]
 
        #Finding angle in degrees
        angle=math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2)  #(in radians)
        degAngle=math.degrees(angle) 
        if degAngle<0:
            degAngle +=360
    
        # print(degAngle)

        #draw the landmarks
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.line(img,(x2,y2),(x3,y3),(255,0,0),3)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255),2)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255),2)
            cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,0,255),2)
            # cv2.putText(img,f'{int(degAngle)} deg',(x2+34,y2+50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
        return degAngle

def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture('videos/1.mp4')
    detector=poseDetector()
    while True:
        success, img=cap.read()
        img=detector.findPose(img)
        lmList=detector.getPosition(img)
        img=detector.findAngle(img,11,13,15)
        print(lmList) # will the return the position of all landmarks in the pose
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()