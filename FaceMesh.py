#there are 468 different landmarks on a face in total
import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('videos/2_1.mp4')
pTime=0

mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
faceMesh=mpFaceMesh.FaceMesh()   #mediapipe FaceMesh object 
drawSpec=mpDraw.DrawingSpec(thickness=2, circle_radius=2)

while True:
    success,img=cap.read()
    img=cv2.resize(img,(790,670),fx=0.5,fy=0.5)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for singleFacelm in results.multi_face_landmarks:   #landmark of one face
            mpDraw.draw_landmarks(img,singleFacelm, FACEMESH_TESSELATION, drawSpec, drawSpec)

            for id,lm in enumerate(singleFacelm.landmark):  
                # print(lm)
                ih, iw, ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'fps: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("images",img)
    cv2.waitKey(1)

    