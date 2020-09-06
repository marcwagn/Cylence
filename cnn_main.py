"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.09.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   CNN main script
########################################################
"""

import numpy as np
import cv2
import glob
import os
import argparse

from tqdm import tqdm

from cnn_detection.image_segmentation_keras.keras_segmentation.predict import model_from_checkpoint_path
from cnn_detection.utils import predictTiled
from quantification.quantification import quantify

#define input parameters
parser = argparse.ArgumentParser(description='CNN based analysis')
parser.add_argument('--inPath', required=True, help='input folder')
parser.add_argument('--outPath', required=True, help='output folder')
args = parser.parse_args()

#load model
model = model_from_checkpoint_path("cnn_detection/model/vgg_unet_cross_aug_ep100_51_100/vgg_unet_1")

#load input images from input folder
img_name_arr = [os.path.basename(x) for x in glob.glob(args.inPath+'/*')]

for img_name in img_name_arr:
    img = cv2.imread(args.inPath+'/'+img_name)
    pred = predictTiled(img,model)

    num_filaments, num_infected_filaments = quantify(img,pred)

    #create overlay
    overlay = cv2.addWeighted(pred, 0.1,img, 0.9,0,dtype = cv2.CV_32F)

    #add text
    res = cv2.putText(overlay, 'num. living filaments = ' + str(num_filaments), 
                      (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 
    res = cv2.putText(overlay, 'num. infected filaments = ' + str(num_infected_filaments), 
                      (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 

    cv2.imwrite(args.outPath+'/'+os.path.splitext(img_name)[0] + '.png', overlay)