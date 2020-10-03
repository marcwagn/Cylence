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
import os
import tifffile as tifi
import numpy as np
import cv2
import glob
import argparse
import configparser
import csv

from tqdm import tqdm

from cnn_detection.image_segmentation_keras.keras_segmentation.predict import model_from_checkpoint_path
from cnn_detection.utils import predictTiled
from quantification.quantification import quantify

#define input parameters
parser = argparse.ArgumentParser(description='CNN based analysis')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--inPath', help='input folder')
group.add_argument('--inFile', help='input file')
parser.add_argument('--predPath', required=True, help='output folder prediction')
parser.add_argument('--outPath', required=True, help='output folder analysis')
args = parser.parse_args()

#load model
model = model_from_checkpoint_path("cnn_detection/model/vgg_unet_cross_aug_ep130/vgg_unet_cross")

#load input images from input folder
if (args.inFile == None):
    img_name_arr = [os.path.basename(x) for x in glob.glob(args.inPath+'/*')]
    inPath = args.inPath
else:
    img_name_arr = [os.path.basename(args.inFile)]
    inPath = os.path.dirname(args.inFile)

for img_name in img_name_arr:
    print('Read image: ' + img_name)
    img = tifi.imread(inPath+'/'+img_name)
    img = img[:,:,::-1].copy()

    print('Create segmentation map: ' + img_name)
    pred = predictTiled(img,model)

    print('Save segmentation map: ' + img_name)
    with tifi.TiffWriter(args.predPath+'/'+os.path.splitext(img_name)[0] + '.tif') as tif:
         options = dict(tile=(256, 256), compression='zlib')
         tif.write(pred[:,:,::-1], subifds=2, **options)
         tif.write(pred[::2,::2,::-1], subfiletype=1, **options)
         tif.write(pred[::4,::4,::-1], subfiletype=1, **options)
    #tifi.imwrite(args.predPath+'/'+os.path.splitext(img_name)[0] + '.tif', pred[:,:,::-1])

    print('Quantify prevalence: ' + img_name)
    #read quantification config file
    config = configparser.ConfigParser()
    config.read('quantification/config.ini')
    para = config['hyperparameter']

    num_filaments, num_infected_filaments = quantify(img, pred, para)
    if num_filaments == 0:
        prev_of_infection = 0
    else:
        prev_of_infection = num_infected_filaments / num_filaments

    #create overlay
    overlay = cv2.addWeighted(pred, 0.1,img, 0.9,0)

    #add text
    res = cv2.putText(overlay, 'num. detected filaments = ' + str(num_filaments), 
                      (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 
    res = cv2.putText(overlay, 'num. infected filaments = ' + str(num_infected_filaments), 
                      (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 

    print('Save quantification: ' + img_name)
    with tifi.TiffWriter(args.outPath+'/'+os.path.splitext(img_name)[0] + '.tif') as tif:
         options = dict(tile=(256, 256), compression='zlib')
         tif.write(overlay[:,:,::-1], subifds=2, **options)
         tif.write(overlay[::2,::2,::-1], subfiletype=1, **options)
         tif.write(overlay[::4,::4,::-1], subfiletype=1, **options)
    #tifi.imwrite(args.outPath+'/'+os.path.splitext(img_name)[0] + '.tif', overlay[:,:,::-1])

    with open(args.outPath+'/'+os.path.splitext(img_name)[0] + '.csv', mode='w') as stat_file:
        stat_writer = csv.writer(stat_file, 
                                 delimiter=',', 
                                 quotechar='"', 
                                 quoting=csv.QUOTE_MINIMAL)
        stat_writer.writerow(['file_name', 'detected_filaments', 'infected_filaments', 'PoI'])
        stat_writer.writerow([os.path.splitext(img_name)[0],
                              num_filaments, 
                              num_infected_filaments,
                              prev_of_infection])
