import cv2
import glob
import numpy as np

"""
This module takes in the photos from either cameras stores it and process it to return the intrinsic and extrinsic camera properties and matrices.
"""

vid_c1= cv2.VideoCapture(1)
vid_c2= cv2.VideoCapture(2)
s1, f1 =vid_c1.read()
s2, f2= vid_c2.read()
i, j, min_frame=0, 0, 10

while i<50:
    s1, f1= vid_c1.read()
    s2, f2= vid_c2.read() #adjusting to exposure
    i+=1
 
while s1  and s2 and j<10:
    cv2.imshow("c1", f1)
    cv2.imshow("c2", f2)
    cv2.waitKey(1)
    if cv2.waitKey(1)== ord("s"):
        print("---saving image---: ", j+1)
        cv2.imwrite(f"caliberation/c1/{j+1}.png", f1)
        cv2.imwrite(f"caliberation/c2/{j+1}.png", f2)
        j+=1
    s1, f1= vid_c1.read()
    s2, f2= vid_c2.read()
    
objpts, imgpts1, imgpts2= [], [], []  
    
criteria= (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
nr, nc= 6, 9 #no. rows and columns excluding 1 block from all sides
scale= 15 #mm side length

objp = np.zeros((nr*nc,3), np.float32)
objp[:,:2] = np.mgrid[0:nr,0:nc].T.reshape(-1,2)
objp = scale* objp

ims1_names, ims2_names= sorted(glob.glob("caliberation/c1/*.png")), sorted(glob.glob("caliberation/c2/*.png"))

for im1_name, im2_name in zip(ims1_names, ims2_names):
    im1= cv2.imread(im1_name, 1)
    im2= cv2.imread(im2_name, 1)
    im1_g= cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_g= cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    ret1, corner1= cv2.findChessboardCorners(im1_g, (nr, nc), None)
    ret2, corner2= cv2.findChessboardCorners(im2_g, (nr, nc), None)
    
    if ret1 and ret2:
        objpts.append(objp)
        corner1= cv2.cornerSubPix(im1_g, corner1, (11, 11), (-1,-1), criteria)
        imgpts1.append(corner1)
        
        corner2= cv2.cornerSubPix(im2_g, corner2, (11, 11), (-1,-1), criteria)
        imgpts2.append(corner2)
        
        cv2.drawChessboardCorners(im1, (nr, nc), corner1, ret1)
        cv2.imshow("im1", im1)
        
        cv2.drawChessboardCorners(im2, (nr, nc), corner2, ret2)
        cv2.imshow("im2", im2)
        
        cv2.waitKey(1000)
        
cv2.destroyAllWindows()

# =============================================================================
# Camera caliberation- Individual
# =============================================================================

h1, w1, c1= im1.shape
f1_size= (w1, h1)
ret1, cameraMatrix1, dist1, rvecs1, tvecs1= cv2.calibrateCamera(objpts, imgpts1, f1_size, None, None)
print("RMSE c1: ", ret1) #for good caliberation thish shd be less than 1 but acceptable till 3
newCameraMatrix1, roi_1= cv2.getOptimalNewCameraMatrix(cameraMatrix1, dist1, f1_size, 1, f1_size)

h2, w2, c2= im2.shape
f2_size= (w2, h2)
ret2, cameraMatrix2, dist2, rvecs2, tvecs2= cv2.calibrateCamera(objpts, imgpts2, f2_size, None, None)
print("RMSE c2: ", ret2) #for good caliberation thish shd be less than 1 but acceptable till 3
newCameraMatrix2, roi_2= cv2.getOptimalNewCameraMatrix(cameraMatrix2, dist2, f2_size, 1, f2_size)

# =============================================================================
# Camera caliberation- Stereo
# =============================================================================

flags=0
flags |=cv2.CALIB_FIX_INTRINSIC
criteria_stereo= (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
ret_stereo,  newCameraMatrix1, dist1, newCameraMatrix2, dist2, rot, trans, essentialMatrix, fundamentalMatrix= cv2.stereoCalibrate(objpts, imgpts1, imgpts2, newCameraMatrix1, dist1, newCameraMatrix2, dist2, im1_g.shape[::-1], criteria_stereo, flags)

print("RMSE stereo: ", ret_stereo)   
# =============================================================================
# Stereo rectification        
# =============================================================================

rectifyScale=1
rect1, rect2, projMatrix1, projMatrix2, Q, roi_1, roi_2= cv2.stereoRectify(newCameraMatrix1, dist1, newCameraMatrix2, dist2, im1_g.shape[::-1], rot, trans, rectifyScale, (0,0))

stereoMap1= cv2.initUndistortRectifyMap(newCameraMatrix1, dist1, rect1, projMatrix1, im1_g.shape[::-1], cv2.CV_16SC2)
stereoMap2= cv2.initUndistortRectifyMap(newCameraMatrix2, dist2, rect2, projMatrix2, im2_g.shape[::-1], cv2.CV_16SC2)

print("saving parameters!!")
cv_file= cv2.FileStorage("stereoMap.xml", cv2.FILE_STORAGE_WRITE)

cv_file.write("stereoMap1_x", stereoMap1[0])
cv_file.write("stereoMap1_y", stereoMap1[1])
cv_file.write("stereoMap2_x", stereoMap2[0])
cv_file.write("stereoMap2_y", stereoMap2[1])

cv_file.release()
vid_c1.release()
vid_c2.release()
