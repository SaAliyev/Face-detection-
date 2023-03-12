import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, refineLms= False, minDetectionCon= 0.5, minTrackCon= 0.5):
        self.staticMode= False
        self.maxFaces= maxFaces
        self.refineLms= False
        self.minDetectionCon= minDetectionCon
        self.minTrackCon= 0.5
        self.mpDraw= mp.solutions.drawing_utils
        self.mpFaceMesh= mp.solutions.face_mesh
        self.faceMesh= self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLms, self.minDetectionCon, self.minTrackCon)
        self.drawSpec= self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,img, draw=True):
        self.imgRGB= cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results= self.faceMesh.process(self.imgRGB)
        faces= []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec, self.drawSpec)
                face= []
                for id,lm in enumerate(faceLms.landmark):
                        #print(lm)
                        ih ,iw, ic= img.shape
                        x, y= int(lm.x*iw), int(lm.y* ih)
                        #print(id,x,y)
                        cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_PLAIN, 0.7, (0,255,0), 1)
                        face.append([x,y])
                faces.append(face)
        return img, faces
    

def main():
    cap= cv.VideoCapture("Videos/6.mp4")
    pTime= 0
    detector= FaceMeshDetector()
    while True:
        success, img= cap.read()
        img, faces= detector.findFaceMesh(img, draw=False)
        if len(faces)!= 0:
            print(faces[0])
        cTime= time.time()
        fps= 1/(cTime-pTime)
        pTime= cTime
        cv.putText(img, f'FPS:{int(fps)}', (20,70), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__== '__main__':
    main()