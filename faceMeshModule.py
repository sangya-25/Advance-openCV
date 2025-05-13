from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
import cv2
import mediapipe as mp
import time
class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTackingCon=0.5):
        self.staticMode=staticMode,
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackingCon=minTackingCon
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.FACEMESH_TESSELATION = FACEMESH_TESSELATION
        self.faceMesh=self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
        )  #mediapipe FaceMesh object 
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for singleFacelm in self.results.multi_face_landmarks:   #landmark of one face
                if draw:
                    self.mpDraw.draw_landmarks(img,singleFacelm, FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
        return img
    
    def findAllLandmarks(self, img):
        faces=[]
        if self.results.multi_face_landmarks:
            for singleFacelm in self.results.multi_face_landmarks:
                lmList=[]
                for id,lm in enumerate(singleFacelm.landmark):  
                    # print(lm)
                    ih, iw, ic=img.shape
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    lmList.append([id,x,y])
                    # print(id,x,y)
                faces.append(lmList) #to get the number of faces detected!
        return img,faces

def main():
    cap=cv2.VideoCapture('videos/2_6.mp4')
    pTime=0
    detector=FaceMeshDetector()
    while True:
        success,img=cap.read()
        # img=cv2.resize(img,(790,670),fx=0.5,fy=0.5)
        img=detector.findFaceMesh(img)
        img,faces=detector.findAllLandmarks(img)
        if len(faces)!=0:
            print(len(faces))
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,f'fps: {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("images",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
