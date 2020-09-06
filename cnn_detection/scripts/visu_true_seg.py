import sys
sys.path.insert(1, '/home/marc/Dropbox/WiSe1920/MA_thesis/03_KERAS_detection/image-segmentation-keras')

import numpy as np
import cv2
import glob
import os

from keras_segmentation.predict import visualize_segmentation

if __name__ == "__main__":
    path = "../dataset/raw/annotations/"
    img_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(path + "*.png")]
    
    for img_name in img_name_arr:
        ann = cv2.imread(path+img_name+".png")[:,:,2]
        #correct labeling
        ann[ann == 1] = 0
        ann[ann == 2] = 1
        ann[ann == 3] = 2

        seg = visualize_segmentation(ann,n_classes=3,colors=[(0,255,0),(0,0,255),(255,0,0)])

        cv2.imwrite("../dataset/raw/annotation_color/{}.png".format(img_name), seg)


