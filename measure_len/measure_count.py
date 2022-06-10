import numpy as np
import cv2
import os 


# predetermined intrinsic param
folder = "0531_8mp_fisheye_forCalib"
os.chdir(f"../{folder}")
params = np.load("0606_rectify_param.npz")
cameraMatrix = params['cm1']
P1 = params['P1']
Q = params['Q']
print(Q)


def findWorldCoord(img_coord_left, img_coord_right):
    x, y = img_coord_left
    d = img_coord_left[0] - img_coord_right[0]
    # print(x, y, d); exit(0)
    homg_coord = Q.dot(np.array([x, y, d, 1.0]))
    coord = homg_coord / homg_coord[3]
    return coord[:-1]


subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def subPixelAccuracy(img, coords):
    cv2.cornerSubPix(img,coords,(11,11),(-1,-1),subpix_criteria)
    return coords

# Point 1
img_coord_left = np.array(
    [[1083, 1410],          # p1
    [2219, 1494]],          # p2
    dtype=np.float32)
img_coord_right = np.array(
    [[866, 1410],          # p1
    [1966, 1494]],          # p2
    dtype=np.float32)

imgL = cv2.imread('0606_test/rectify_02_left.jpg', 0)
imgR = cv2.imread('0606_test/rectify_02_right.jpg', 0)
subPixelAccuracy(imgL, img_coord_left)
subPixelAccuracy(imgR, img_coord_right)
print(img_coord_left)
print(img_coord_right)

coord1 = findWorldCoord(img_coord_left[0], img_coord_right[0])
coord2 = findWorldCoord(img_coord_left[1], img_coord_right[1])

print(cv2.norm(coord1, coord2))


# original size
# 02 - 1513.3067552683322
# 03 - 1513.6819207017713
# expand size to maintian pixel level detail
# 02 - 1512.9180161022646
# subpix + expand
# 02 - 1510.7306663429006

# down edge 1504.76601788457