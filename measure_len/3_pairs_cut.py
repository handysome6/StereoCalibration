import cv2
import os

# Global variables preset
total_photos =54
photo_width = 8112
photo_height = 3040
img_width = 4056
img_height = 3040
photo_counter = 0


# Main pair cut cycle
if (os.path.isdir("./pairs")==False):
    os.makedirs("./pairs")
while photo_counter != total_photos:
    photo_counter +=1
    filename = './scenes/'+'sbs_'+str(photo_counter).zfill(2) + '.jpg'
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
