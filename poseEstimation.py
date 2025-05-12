import cv2 
import mediapipe as mp 
import time

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpdraw=mp.solutions.drawing_utils

pTime=0
cTime=0
cap=cv2.VideoCapture('videos/3.mp4')
while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if(results.pose_landmarks):
        landmark_spec = mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4) 
        connection_spec = mpdraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        mpdraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS, landmark_spec, connection_spec)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)  #to get the acutal pixel values in integer and not in ratio
            print(id,cx,cy)
            #to ensure we are getting the right location for each landmarks, drawing circle over the positions to verify
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


