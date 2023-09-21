import cv2
import numpy as np

"""
This function gets us the 3d coordinates when the center of the boxes and the camera parameters are provided.
"""

def find_dist(pt1, pt2, f1, f2, baseline, f, alpha):
    h1, w1, ch1= f1.shape
    h2, w2, ch2= f2.shape
    x0=baseline/2
    y0=0 #we consider both the cameras in the same 0 height 
    
    if w1==w2:
        f_pixel= (w1*0.5)/(np.tan(alpha*0.5*np.pi/180)) #converting f from mm to pixels
    else: print("frames have different widths")
    
    x1, x2= pt1[0], pt2[0]
    y1, y2= pt1[1], pt2[1]
    disparity= x1-x2
    dist_z= ((baseline*f_pixel)/disparity)
    dist_x= (x1-320)*dist_z/f_pixel
    dist_x= dist_x+ x0 #measuring from the center of the stereo system
    dist_y= y1*dist_z/f_pixel
    dist_y= dist_y- y0
    return dist_x, dist_y, dist_z
