import cv2
import mediapipe as mp
import time
class faceDetector():
    def __init__(self,mindetectionCon=0.5):
        self.minDetectionCon=mindetectionCon
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection()

    def getBbox(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                h,w,c=img.shape
                bboxC=detection.location_data.relative_bounding_box  #bounding box coming from the class
                bbox=int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img=self.fancyDraw(img,bbox)#to draw the rectangle across face by using bbox parameters for height, widht and x, y values with styling
                    cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0]-130,bbox[1]+30),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
        return img, bboxs

    def fancyDraw(self , img, bbox, l=30, t=5):  #length and thickness
        x,y,w,h=bbox 
        x1,y1 = x+w, y+h
        cv2.rectangle(img,bbox,(255,0,255),1)  
        #this is the fancy draw i.e line for the top left (x,y) as origin
        cv2.line(img,(x,y),(x+l, y),(255,0,255),t)
        cv2.line(img,(x,y),(x, y+l),(255,0,255),t)

        #for top right i.e x1,y as origin
        cv2.line(img,(x1,y),(x1-l, y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1, y+l),(255,0,255),t)

        #for the bottom left i.e. x, y1 as origin
        cv2.line(img,(x,y1),(x+l, y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x, y1-l),(255,0,255),t)

        #this is for the bottom right i.e for (x1,y1) as origin
        cv2.line(img,(x1,y1),(x1-l, y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1, y1-l),(255,0,255),t)
        return img

def main():
    cTime=0
    pTime=0
    cap=cv2.VideoCapture('videos/2_2.mp4')
    detector=faceDetector()
    while True:
        success, img=cap.read()
        img, bboxs=detector.getBbox(img)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'fps: {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow("face image", img)
        cv2.waitKey(10)


if __name__=="__main__":
    main()