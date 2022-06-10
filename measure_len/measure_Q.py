import numpy as np
import cv2

# project to 3D using Q flaw with rectified images pair
Q = np.array([
    [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.04068286e+03],
    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.54033621e+03],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.00110616e+03],
    [ 0.00000000e+00,  0.00000000e+00,  4.54827405e-03, -0.00000000e+00]])
u0 = 2.01093445e+03
v0 = 1.55493561e+03


rows = 8
columns = 11
CHECKERBOARD = (rows,columns)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

leftName = '../0526_ensightful_calib/rectify_vanilla/rectify_01_left.jpg'
rightName = '../0526_ensightful_calib/rectify_vanilla/rectify_01_right.jpg'
imgL = cv2.imread(leftName,1)
grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
imgR = cv2.imread(rightName,1)
grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
# Find the chessboard corners
retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
if ((retL == True) and (retR == True)):
    cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),subpix_criteria)
    cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),subpix_criteria)

# print(cornersL[0])
coord1 = [cornersL[0][0][0], cornersL[0][0][1], cornersL[0][0][0]-cornersR[0][0][0]]
point_and_disparity = np.array([[coord1]], dtype=np.float32)
coord1 = cv2.perspectiveTransform(point_and_disparity, Q)

coord2 = [cornersL[1][0][0], cornersL[1][0][1], cornersL[1][0][0]-cornersR[1][0][0]]
point_and_disparity = np.array([[coord2]], dtype=np.float32)
coord2 = cv2.perspectiveTransform(point_and_disparity, Q)
print(cv2.norm(coord1, coord2))

point_and_disparity = np.array([[[3365,871,3365-2050]]], dtype=np.float32)
coord1 = cv2.perspectiveTransform(point_and_disparity, Q)
point_and_disparity = np.array([[[3373,1019, 3373-2054]]], dtype=np.float32)
coord2 = cv2.perspectiveTransform(point_and_disparity, Q)
print(cv2.norm(coord1, coord2))