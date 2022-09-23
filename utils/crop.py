import cv2
import glob

images = glob.glob(..data.original*jpg)

for img in images:
    img = cv2.imread(img)
    cropped_img = img[0:1536, 0:2048]
    cv2.imwrite("img", cropped_img)