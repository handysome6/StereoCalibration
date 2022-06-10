from importlib.machinery import all_suffixes
import numpy as np
import cv2
import os 
from click_coord import ClickImage


# chdir to 
operation_folder = "0610_IMX477_infinity_still"
os.chdir(f"{operation_folder}")


##### change image id here #####
measure_id = 1
imgL = cv2.imread(f'test/rectify_{measure_id}_left.jpg', 0)
imgR = cv2.imread(f'test/rectify_{measure_id}_right.jpg', 0)
# imgL = cv2.imread('rectify_vanilla/rectify_01_left.jpg', 0)
# imgR = cv2.imread('rectify_vanilla/rectify_01_right.jpg', 0)

# predetermined intrinsic param
params = np.load(f"{operation_folder}.npz")
cameraMatrix = params['cm1']
P1 = params['P1']
Q = params['Q']
T = params['T']
f = cameraMatrix[0][0]
u0 = cameraMatrix[0][2]
v0 = cameraMatrix[1][2]
f = P1[0][0]
u0 = P1[0][2]
v0 = P1[1][2]
b = cv2.norm(T)

def findWorldCoord(img_coord_left, img_coord_right):
    x, y = img_coord_left
    d = img_coord_left[0] - img_coord_right[0]
    # print(x, y, d); exit(0)
    homg_coord = Q.dot(np.array([x, y, d, 1.0]))
    coord = homg_coord / homg_coord[3]
    print(coord)
    return coord[:-1]

# By epipolar geometry
def findWorldCoord2(img_coord_left, img_coord_right):
    u1, v1 = img_coord_left
    u2, v2 = img_coord_right
    x = u1 - u0
    y = v1 - v0
    disp = u1-u2
    world_z = b * f / disp
    world_x = world_z / f * x
    world_y = world_z / f * y
    return np.array([world_x, world_y, world_z])


subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.00001)
def subPixelAccuracy(img, coords):
    cv2.cornerSubPix(img,coords,(11,11),(-1,-1),subpix_criteria)
    return coords



left = ClickImage(imgL, 'left')
img_coord_left = left.click_coord()
right = ClickImage(imgR, 'right')
img_coord_right = right.click_coord()

subPixelAccuracy(imgL, img_coord_left)
subPixelAccuracy(imgR, img_coord_right)
# print(img_coord_left)
# print(img_coord_right)


coord1 = findWorldCoord(img_coord_left[0], img_coord_right[0])
coord2 = findWorldCoord(img_coord_left[1], img_coord_right[1])

print(cv2.norm(coord1, coord2))



# display corner
def draw_line_crop(img, point):
    line_thickness = 1
    point = (int(point[0]), int(point[1]))
    cv2.line(img, point, (point[0], 0), (0,0,0), thickness=line_thickness)
    cv2.line(img, point, (0, point[1]), (0,0,0), thickness=line_thickness)
    return \
        img[point[1]-50:point[1]+50,
            point[0]-50:point[0]+50,]
            
l1 = draw_line_crop(imgL, img_coord_left[0])
l2 = draw_line_crop(imgL, img_coord_left[1])
r1 = draw_line_crop(imgR, img_coord_right[0])
r2 = draw_line_crop(imgR, img_coord_right[1])
l = np.hstack([l1,l2])
r = np.hstack([r1,r2])
crop = np.vstack([l,r])
crop = cv2.resize(crop, [400, 400])
cv2.imshow('crop', crop)
key = cv2.waitKey(0)
if key == 27:
   cv2.destroyAllWindows()



# new = 1
# 1.525@1.9m -> 1504
# 1.525@3.0m -> 1482
# 200.0@1.8m -> 200.9,200.99,198.23,200.24
# new = 2

