import numpy as np
import cv2

def calibrate(fname):
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane

  # prepare object points
  nx = 4 #number of inside corners in x
  ny = 4 #number of inside corners in y

  objp = np.zeros((nx*ny,3), np.float32)
  objp[:,:2] =  np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates 

  # Make a list of calibration images

  img = cv2.imread(fname)
  # Convert to grayscale
  res = cv2.resize(img,None,fx=0.2,fy=0.2)

  gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
  # Find the chessboard corners
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

  # If found, draw corners
  if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    #print(ret, mtx, dist,rvecs,tvecs)

    # Draw and display the corners
    #cv2.drawChessboardCorners(res, (nx, ny), corners, ret)
    #cv2.imshow("chessboard", res)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print(mtx.tolist()) #camera calibration

calibrate('./calib_deitado.png')
calibrate('./calib_deitado_2.png')
calibrate('./calib_deitado_3.png')
calibrate('./calib_deitado_4.png')
