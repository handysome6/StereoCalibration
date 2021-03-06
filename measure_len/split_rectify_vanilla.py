import cv2
import os
import numpy as np


# Camera resolution
photo_width = 8112
photo_height = 3040
# Image resolution for processing
img_width = 4056
img_height = 3040
imageSize = (img_width,img_height)


# chdir to 
operation_folder = "0610_IMX477_infinity_still"
os.chdir(f"{operation_folder}")

# predetermined intrinsic param
params = np.load("0610_IMX477_infinity_still.npz")
cm1 = params['cm1']
cd1 = params['cd1']
cm2 = params['cm2']
cd2 = params['cd2']
R1 = params['R1']
P1 = params['P1']
R2 = params['R2']
P2 = params['P2']
newImageSize = params['newImageSize']

# build map
leftMapX, leftMapY = cv2.initUndistortRectifyMap(cm1, cd1, R1, P1, newImageSize, cv2.CV_16SC2)
rightMapX, rightMapY= cv2.initUndistortRectifyMap(cm2, cd2, R2, P2, newImageSize, cv2.CV_16SC2)


# read
os.chdir('test')
print('rectifying test images...')
for i in range(13):
    filename = str(i+1)
    img = f'{filename}.jpg'
    if os.path.isfile(img) == False:
        print (f"No file named {img}"); exit()
    pair_img = cv2.imread(img,-1)

    # split
    imgL = pair_img [0:img_height,0:img_width] #Y+H and X+W
    imgR = pair_img [0:img_height,img_width:photo_width]
    leftName = f'{filename}_left.jpg'
    rightName = f'{filename}_right.jpg'

    # rectify
    imgL = cv2.remap(imgL, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgR, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(f"rectify_{filename}_left.jpg", imgL)
    cv2.imwrite(f"rectify_{filename}_right.jpg",imgR)

print('rectify successful.')