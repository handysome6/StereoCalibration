import cv2
import os
from pathlib import Path


# Global variables preset
photo_width = 8112
photo_height = 3040
img_width = 4056
img_height = 3040
operation_folder = '0610_IMX477_infinity_still'

print(os.getcwd())
# change to operation folder
os.chdir(f"{operation_folder}")
total_photos = len(list(Path('scenes').glob('*.jpg')))


# rename
p = Path('.')
files = [f for f in p.glob('scenes/*.jpg')]
prefix = "scenes/"
for i in range(len(files)):
    original_name = files[i].name
    os.rename(prefix+original_name, prefix+f"sbs_{str(i+1).zfill(2)}.jpg")


# Main pair cut cycle
if (os.path.isdir("pairs/")==False):
    os.makedirs("pairs/")
photo_counter = 0
while photo_counter != total_photos:
    photo_counter +=1
    filename = 'scenes/'+'sbs_'+str(photo_counter).zfill(2) + '.jpg'
    if os.path.isfile(filename) == False:
        print ("No file named "+filename)
        continue
    pair_img = cv2.imread(filename,-1)
    
    #cv2.imshow("ImagePair", pair_img)
    #cv2.waitKey(0)
    imgLeft = pair_img [0:img_height,0:img_width] #Y+H and X+W
    imgRight = pair_img [0:img_height,img_width:photo_width]
    leftName = './pairs/'+str(photo_counter).zfill(2)+'_left.png'
    rightName = './pairs/'+str(photo_counter).zfill(2)+'_right.png'
    cv2.imwrite(leftName, imgLeft)
    cv2.imwrite(rightName, imgRight)
    print ('Pair No '+str(photo_counter)+' saved.')
    
print ('End cycle')
