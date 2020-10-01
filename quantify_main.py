"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.09.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   quantification script
########################################################
"""
import glob
import cv2
import os
import numpy as np
import argparse
import configparser
import tifffile as tifi

from skimage.morphology import skeletonize
from scipy import signal

import networkx as nx

from quantification.connCompAnalysis import ConnCompAnalysis
from quantification.quantification import quantify

#define input parameters
parser = argparse.ArgumentParser(description='CNN based analysis')
parser.add_argument('--imgFile', help='input file')
parser.add_argument('--predFile', required=True, help='prediction file')
parser.add_argument('--outPath', required=True, help='output folder analysis')
args = parser.parse_args()

#load input images from input folder
img_name = os.path.basename(args.imgFile)


#load data
print('Read image: ' + img_name)
img = tifi.imread(args.imgFile)

print('Read prediction: ' + img_name)
pred = tifi.imread(args.predFile)

print('Quantify prevalence: ' + img_name)
#read quantification config file
config = configparser.ConfigParser()
config.read('quantification/config.ini')
para = config['hyperparameter']

num_filaments, num_infected_filaments = quantify(img, pred, para)

#overlay segmentation
res = cv2.addWeighted(pred, 0.1,img, 0.9,0)
#add text
res = cv2.putText(res, 'num. living filaments = ' + str(num_filaments), 
                  (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                  1, (0,0,0), 2, cv2.LINE_AA) 
res = cv2.putText(res, 'num. infected filaments = ' + str(num_infected_filaments), 
                  (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                  1, (0,0,0), 2, cv2.LINE_AA) 

tifi.imwrite(args.outPath+"{}".format(img_name),res)