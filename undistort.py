import cv2
"""
This function helps remove distortions that maybe present in the image due to the camera/ camera lens.
"""
cv_file= cv2.FileStorage()
cv_file.open("stereoMap.xml", cv2.FileStorage_READ)

stereoMap1_x= cv_file.getNode("stereoMap1_x").mat()
stereoMap1_y= cv_file.getNode("stereoMap1_y").mat()
stereoMap2_x= cv_file.getNode("stereoMap2_x").mat()
stereoMap2_y= cv_file.getNode("stereoMap2_y").mat()

def undistort(f1, f2):
    undistort_f1= cv2.remap(f1, stereoMap1_x, stereoMap1_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistort_f2= cv2.remap(f2, stereoMap2_x, stereoMap2_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return undistort_f1, undistort_f2
