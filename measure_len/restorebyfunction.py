import numpy as np
import cv2
import os
from statistics import mean, stdev
from tqdm.auto import tqdm


# chdir to 
operation_folder = "0608_IMX477_infinity"
os.chdir(f"../{operation_folder}")

# predetermined intrinsic param
params = np.load("0608_IMX477_infinity.npz")
cameraMatrix = params['cm1']
P1 = params['P1']
Q = params['Q']
# print(Q); exit()


def findWorldCoord(img_coord_left, img_coord_right):
    x, y = img_coord_left[0]
    d = img_coord_left[0][0] - img_coord_right[0][0]
    # print(x, y, d); exit(0)
    homg_coord = Q.dot(np.array([x, y, d, 1.0]))
    coord = homg_coord / homg_coord[3]
    return coord[:-1]


rows = 8
columns = 11
CHECKERBOARD = (rows,columns)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


for id in (range(30)):
    image_id = str(id+1).zfill(2)
    leftName = f'rectify_vanilla/rectify_{image_id}_left.jpg'
    rightName = f'rectify_vanilla/rectify_{image_id}_right.jpg'
    imgL = cv2.imread(leftName,1)
    if imgL is None:
        continue
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    imgR = cv2.imread(rightName,1)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
    if ((retL == True) and (retR == True)):
        cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),subpix_criteria)
        cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),subpix_criteria)
    else:
        print(f"Pair {image_id}, no chessboard found")
        continue
    
    # find the 
    edges = []
    for i in range(7):
        # select two points
        point_id_1 = i
        point_id_2 = i+1

        img_coord_1 = [cornersL[point_id_1], cornersR[point_id_1]]
        img_coord_2 = [cornersL[point_id_2], cornersR[point_id_2]]


        coord1 = findWorldCoord(*img_coord_1)
        coord2 = findWorldCoord(*img_coord_2)

        distance = cv2.norm(coord1, coord2)
        if distance < 100:
            edges.append(distance)
            # print(distance)
    print(f"Pair {image_id}, Mean edge len: {mean(edges)}")
    
# print(f'''
# Over all performance:
# Max: {max(edges)}; Min: {min(edges)}
# Mean: {mean(edges)}; Stdev: {stdev(edges)}
# ''')