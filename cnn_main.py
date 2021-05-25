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
import numpy as np
import pandas as pd
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
parser.add_argument('--predPath', help='output folder prediction')
parser.add_argument('--outPath', help='output folder analysis')
args = parser.parse_args()

#load model
model = model_from_checkpoint_path("cnn_detection/model/vgg_unet_cross/vgg_unet_cross")

#input: single file
if (args.inFile is not None):
    img_name_arr = [os.path.basename(args.inFile)]
    inPath = os.path.dirname(args.inFile)

    id_list = [os.path.splitext(img_name)[0] for img_name in img_name_arr]
    stats = pd.DataFrame(id_list, columns =['ID'])
    stats_file_found = False

#input: folder
if (args.inPath is not None):
    img_name_arr = [os.path.basename(x) for x in glob.glob(args.inPath+'/*.png')]
    inPath = args.inPath

    if (os.path.isfile(args.inPath + 'stats.csv')):
        stats = pd.read_csv(args.inPath + 'stats.csv', sep=';', header = 0)
        stats_file_found = True
    else:
        id_list = [os.path.splitext(img_name)[0] for img_name in img_name_arr]
        stats = pd.DataFrame(id_list, columns =['ID'])
        stats_file_found = False

stats['COUNT_INF'] = 0
stats['COUNT_TOTAL'] = 0
stats = stats.set_index('ID')


for img_name in tqdm(img_name_arr):
    print('Read image: ' + img_name)
    img = cv2.imread(inPath+'/'+img_name)

    #Step 1: create segmentation map
    pred = predictTiled(img,model)

    if(args.predPath is not None):
        cv2.imwrite(args.predPath+'/'+os.path.splitext(img_name)[0] + '.png', pred)

    
    #Step 2: Quantify Filaments/Infection
    config = configparser.ConfigParser()
    config.read('quantification/config.ini')
    para = config['hyperparameter']

    num_filaments, num_infected_filaments = quantify(img, pred, para)
    stats.loc[os.path.splitext(img_name)[0],'COUNT_INF'] = num_infected_filaments
    stats.loc[os.path.splitext(img_name)[0], 'COUNT_TOTAL'] = num_filaments

    check_img = True
    if (stats_file_found):
        if ( stats.loc[os.path.splitext(img_name)[0],'COUNT_INF'] == stats.loc[os.path.splitext(img_name)[0],'NUM_INF'] or
             stats.loc[os.path.splitext(img_name)[0],'COUNT_TOTAL'] == stats.loc[os.path.splitext(img_name)[0],'NUM_TOTAL']):
            check_img = False

    if (args.outPath is not None and check_img == True):
        overlay = cv2.addWeighted(pred, 0.1,img, 0.9,0)

        res = cv2.putText(overlay, 'num. detected filaments = ' + str(num_filaments), 
                          (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0,0,0), 2, cv2.LINE_AA) 
        res = cv2.putText(overlay, 'num. infected filaments = ' + str(num_infected_filaments), 
                          (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0,0,0), 2, cv2.LINE_AA) 

        cv2.imwrite(args.outPath+'/'+os.path.splitext(img_name)[0] + '.png', overlay)


#Step 3: write Stats
if (args.inPath is not None):
    stats.to_csv(args.inPath + '/results.csv', sep=';')

if (args.inFile is not None):
    print("filaments = {}; infected = {}".format(stats.loc[os.path.splitext(img_name)[0], 'COUNT_TOTAL'],
                                                    stats.loc[os.path.splitext(img_name)[0],'COUNT_INF']))