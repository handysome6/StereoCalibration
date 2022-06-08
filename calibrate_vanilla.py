import os
import cv2
import numpy as np

imageSrc = './scenes'
imageDest='./rectified'

# Global variables preset
total_photos = 30

# Camera resolution
photo_width = 8112
photo_height = 3040

# Image resolution for processing
img_width = 4056
img_height = 3040
imageSize = (img_width,img_height)

# Chessboard parameters
rows = 8
columns = 11
CHECKERBOARD = (rows,columns)
square_size = 25            # in mm

# Chessboared coordinates
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size         # stereoCalibrate() export R and T in this scale

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []              # 3d point in real world space
imgpointsLeft = []          # 2d points in image plane.
imgpointsRight = []         # 2d points in image plane.


try:
    npz_file = np.load('./calibration_data/chessboard_calib.npz')
    objpoints = npz_file['objpoints']
    imgpointsLeft = npz_file['imgpointsLeft']
    imgpointsRight = npz_file['imgpointsRight']
except:
    def chessboard_calib():
        photo_counter = 0
        print ('Main cycle start')

        while photo_counter != total_photos:
            photo_counter = photo_counter + 1
            print ('Import pair No ' + str(photo_counter))
            leftName = './pairs/'+str(photo_counter).zfill(2)+'_left.png'
            rightName = './pairs/'+str(photo_counter).zfill(2)+'_right.png'
            leftExists = os.path.isfile(leftName)
            rightExists = os.path.isfile(rightName)
            
            # If pair has no left or right image - exit
            if ((leftExists == False) or (rightExists == False)) and (leftExists != rightExists):
                print ("Pair No ", photo_counter, "has only one image! Left:", leftExists, " Right:", rightExists )
                continue 
            
            # else find corners
            imgL = cv2.imread(leftName,1)
            grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
            imgR = cv2.imread(rightName,1)
            grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK
            retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags=flags)
            retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags=flags)

            # Refine corners and add to array for processing
            if ((retL == True) and (retR == True)):
                objpoints.append(objp)
                cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),subpix_criteria)
                imgpointsLeft.append(cornersL)
                cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),subpix_criteria)
                imgpointsRight.append(cornersR)
                continue

            else:
                print ("Pair No", photo_counter, "ignored, as no chessboard found" )
                continue

        print ('End cycle')

        # Now we'll write our results to the file for the future use
        if (os.path.isdir('./calibration_data/')==False):
            os.makedirs('./calibration_data/')
        np.savez('./calibration_data/chessboard_calib.npz',
            objpoints = objpoints, 
            imgpointsLeft = imgpointsLeft, imgpointsRight = imgpointsRight,)

    chessboard_calib()


############# Filter outliers ###############
outliers_id = []       # result of perViewError <- stereoCalibrateExtended()
inliers =   [True if id not in outliers_id 
            else False
            for id in range(total_photos)]
objpoints = objpoints[inliers]
imgpointsLeft = imgpointsLeft[inliers]
imgpointsRight = imgpointsRight[inliers]


########## Calibrate Single Camera ##########
# ret               -> RMSE, Root Mean Square Error
# cameraMatrix      -> K, Intrinsic Params including fx, fy, u0, v0
# distCoeff         -> Distortion params
# rvecs             -> Rotation Vectors, maps world coord to image coord
# tvecs             -> Translation Vecs
ret1, cameraMatrix1, distCoeffs1, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpointsLeft, imageSize, None, None)
ret2, cameraMatrix2, distCoeffs2, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpointsRight, imageSize, None, None)
print(f'''
=============== Single Camera Calib ===============
--------------Left Camera---------------
RMS (Root Mean Square Error): {ret1}
K (Intrinsic Params): \n{cameraMatrix1}
''', end='')
print(f'''
--------------Right Camera-------------
RMS (Root Mean Square Error): {ret2}
K (Intrinsic Params): \n{cameraMatrix2}
''')


calib_criteria = (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags = cv2.CALIB_USE_INTRINSIC_GUESS

########## Calibrate Dual Camera Using Stereo Algo ##########
# This can be replaced by ~Extended() function; pass R, T; get perViewError.
# RMS -> error in pixel
# cm1,2 -> camera matrix
# cd1,2 -> camera distortion
# R, T -> rotation and translation between two cameras
# E, F -> esstential and fundamental matrix
# perViewError -> RMS for every image pair
RMS, cm1, cd1, cm2, cd2, R, T, E, F = \
    cv2.stereoCalibrate(            
        objpoints, 
        imgpointsLeft, imgpointsRight, 
        cameraMatrix1, distCoeffs1, 
        cameraMatrix2, distCoeffs2, 
        imageSize, 
        # R, T, 
        criteria=calib_criteria, 
        flags=flags
    )
print(f'''
================ Dual Camera Calib ================
RMS: {RMS}
----------Left Cam----------
K (Intrinsic Params): \n{cm1}
Distortion Coefficient:\n{cd1}
----------Right Cam---------
K (Intrinsic Params): \n{cm2}
Distortion Coefficient:\n{cd2}
------Extrinsic Params------
R: \n{R}
T: \n{T}
''', end='')
# print(perViewError)


# calculate rectify matrices using calibration param
R1, R2, P1, P2, Q, ROI1, ROI2 = \
    cv2.stereoRectify(
        cm1, cd1, 
        cm2, cd2, 
        imageSize, R, T
    )


# save all parameters
save_file = '0530_rectify_param'
print("Saving rectify param to ../{save_file}.npz")
np.savez(f'../0530_rectify_param.npz',
    cm1 = cm1, cd1 = cd1, 
    cm2 = cm2, cd2 = cd2,
    R = R, T = T, 
    R1 = R1, P1 = P1, 
    R2 = R2, P2 = P2,
    Q = Q,
    ROI1 = ROI1, ROI2 = ROI2,
    )


# #map and save
# leftMapX, leftMapY = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_16SC2)
# rightMapX, rightMapY= cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_16SC2)
# imageDest = './rectify_vanilla'
# if (os.path.isdir(imageDest)==False):
#     os.makedirs(imageDest)
# for i in range(total_photos):
#     print("Rectifying photo id:", i)
#     if i in outliers_id:
#         continue
#     imgL = cv2.imread('pairs/'+str(i+1).zfill(2)+'_left.png')
#     imgR = cv2.imread('pairs/'+str(i+1).zfill(2)+'_right.png')
#     imgL = cv2.remap(imgL, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     imgR = cv2.remap(imgR, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#     cv2.imwrite(imageDest+"/"+"rectify_"+str(i+1).zfill(2)+"_left.jpg",imgL)
#     cv2.imwrite(imageDest+"/"+"rectify_"+str(i+1).zfill(2)+"_right.jpg",imgR)
