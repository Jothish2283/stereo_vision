import cv2
import triangulation
import undistort_frame
import mediapipe as mp
import time

mp_facedetector= mp.solutions.face_detection
mp_draw= mp.solutions.drawing_utils

vid_c1=cv2.VideoCapture(1)
vid_c2=cv2.VideoCapture(2)



print("vid_c1: ", vid_c1.isOpened(),"\nvid_c2: ", vid_c2.isOpened())
w1, h1= vid_c1.get(cv2.CAP_PROP_FRAME_WIDTH), vid_c1.get(cv2.CAP_PROP_FRAME_HEIGHT)
w2, h2= vid_c2.get(cv2.CAP_PROP_FRAME_WIDTH), vid_c2.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("\n-------\n", w1, w2, h1, h2)

s1, f1=vid_c1.read()
s2, f2=vid_c2.read()

f_mm= 4
d= 7 #distance between cameras in cm
alpha= 55 #in degree

with mp_facedetector.FaceDetection(min_detection_confidence= 0.9) as face_detector:
    while s1 and s2:
        # f1, f2= undistort_frame.undistort(f1, f2)
        start_t= time.time()
        f1= cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f2= cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        
        r1= face_detector.process(f1)
        r2= face_detector.process(f2)
        
        f1= cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        f2= cv2.cvtColor(f2, cv2.COLOR_RGB2BGR)
        
        if r1.detections and r2.detections:
            for (detectionx, detectiony) in zip(r1.detections, r2.detections):
                mp_draw.draw_detection(f1, detectionx)
                mp_draw.draw_detection(f2, detectiony)
                bbox1= detectionx.location_data.relative_bounding_box
                bbox2= detectiony.location_data.relative_bounding_box
                bbox1= int(bbox1.xmin*w1), int(bbox1.ymin*h1), int(bbox1.width*w1), int(bbox1.height*h1)
                bbox2= int(bbox2.xmin*w2), int(bbox2.ymin*h2), int(bbox2.width*w2), int(bbox2.height*h2)
                c1= (bbox1[0]+bbox1[2]/2, bbox1[1]+bbox1[3]/2)
                c2= (bbox2[0]+bbox2[2]/2, bbox2[1]+bbox2[3]/2)
                cv2.putText(f1, f'score: {detectionx.score[0]*100 :.2f}%', (bbox1[0], bbox1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(f2, f'score: {detectiony.score[0]*100 :.2f}%', (bbox2[0], bbox2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                x, y, z= triangulation.find_dist(c1, c2, f1, f2, d, f_mm, alpha)
                dist= (x**2+y**2+z**2)**0.5
                
                cv2.putText(f1, f"x: {x :.2f}", (bbox1[0], bbox1[1]+bbox1[3] +45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f1, f"y: {y :.2f}", (bbox1[0], bbox1[1]+bbox1[3] +30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f1, f"z: {z :.2f}", (bbox1[0], bbox1[1]+bbox1[3] +15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f1, f"d: {dist :.2f}", (bbox1[0], bbox1[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                cv2.putText(f2, f"x: {x :.2f}", (bbox2[0], bbox2[1]+bbox2[3] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f2, f"y: {y :.2f}", (bbox2[0], bbox2[1]+bbox2[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f2, f"z: {z :.2f}", (bbox2[0], bbox2[1]+bbox2[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(f2, f"d: {dist :.2f}", (bbox2[0], bbox2[1] - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
                cv2.putText(f2, "Tracking lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(f1, "Tracking lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        end_t= time.time()
        duration= end_t-start_t
        fps= int(1/duration)
        cv2.putText(f1, f'FPS: {fps}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.putText(f2, f'FPS: {fps}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)  
        
        output= cv2.hconcat([f2, f1])
        cv2.imshow("output", output)
        cv2.waitKey(1)
        
        if cv2.waitKey(1)==27:
            break
        
        s1, f1= vid_c1.read()
        s2, f2= vid_c2.read()
            
vid_c1.release()
vid_c2.release()
cv2.destroyAllWindows()
