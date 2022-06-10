import cv2
import os
import numpy as np


# Camera resolution
photo_width = 6560
photo_height = 2464
# Image resolution for processing
img_width = 3280
img_height = 2464
imageSize = (img_width,img_height)


folder = "0607_fisheye_near+far"
os.chdir(f"../{folder}")
# read param
params = np.load("0606_rectify_param.npz")
cm1 = params['cm1']
cd1 = params['cd1']
cm2 = params['cm2']
cd2 = params['cd2']
R1 = params['R1']
P1 = params['P1']
R2 = params['R2']
P2 = params['P2']
newImageSize = params['newImageSize']
# new_K1 = params['new_K1']
# new_K2 = params['new_K2']

# build map
leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(cm1, cd1, R1, P1, newImageSize, cv2.CV_16SC2)
rightMapX, rightMapY= cv2.fisheye.initUndistortRectifyMap(cm2, cd2, R2, P2, newImageSize, cv2.CV_16SC2)


# read
os.chdir('test')
filename = '06'
img = f'{filename}.jpg'
if os.path.isfile(img) == False:
    print (f"No file named {img}"); exit()
pair_img = cv2.imread(img,-1)

# split
imgL = pair_img [0:img_height,0:img_width] #Y+H and X+W
imgR = pair_img [0:img_height,img_width:photo_width]
leftName = f'{filename}_left.jpg'
rightName = f'{filename}_right.jpg'
# cv2.imwrite(leftName, imgL)
# cv2.imwrite(rightName, imgR)

# rectify
imgL = cv2.remap(imgL, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
imgR = cv2.remap(imgR, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imwrite(f"rectify_{filename}_left.jpg", imgL)
cv2.imwrite(f"rectify_{filename}_right.jpg",imgR)

