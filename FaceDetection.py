import cv2
import mediapipe as mp
import time

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection()

cTime=0
pTime=0
cap=cv2.VideoCapture('videos/2_6.mp4')
while True:
    success, img=cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)  #a built-in function detect faces and to draw  6 landmarks over the faces
            #print(id,detection)
            #getting the information about the box coordinates of the face
            h,w,c=img.shape
            bboxC=detection.location_data.relative_bounding_box  #bounding box coming from the class
            bbox=int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(img,bbox,(255,0,255),2)  #to draw the rectangle across face by using bbox parameters for height, widht and x, y values
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0]-130,bbox[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'fps: {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("face image", img)
    cv2.waitKey(10)
