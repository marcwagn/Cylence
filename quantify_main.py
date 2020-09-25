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

from skimage.morphology import skeletonize
from scipy import signal

import networkx as nx

from quantification.connCompAnalysis import ConnCompAnalysis

#define input parameters
parser = argparse.ArgumentParser(description='CNN based analysis')
parser.add_argument('--imgPath', required=True, help='image folder')
parser.add_argument('--predPath', required=True, help='prediction folder')
parser.add_argument('--outPath', required=True, help='output folder analysis')
args = parser.parse_args()

#load input images from input folder
img_name_arr = [os.path.basename(x) for x in glob.glob(args.predPath+'/*')]

for img_name in img_name_arr:

    #load data
    print('Read image: ' + img_name)
    img = cv2.imread(args.imgPath+'/'+ os.path.splitext(img_name)[0] + '.tif')

    print('Read prediction: ' + img_name)
    pred = cv2.imread(args.predPath+'/'+img_name)

    #create binary masks
    bin_infection = pred[:,:,2] // 255
    bin_filament = pred[:,:,1] // 255
    bin_background = pred[:,:,0] // 255

    conn_comp = cv2.connectedComponentsWithStats(bin_filament , cv2.CV_32S, connectivity=8)

    #result parameter
    num_filaments = 0
    num_infected_filaments = 0

    #cv2.imshow(img_name+'_old',img)

    #iterate over all connected components
    for idx in range(1,conn_comp[0]):
        #create subimage of CC
        x, y, width, height, _ = conn_comp[2][idx]
        sub_img = img[y:y+height, x:x+width]
        sub_bin_filament = conn_comp[1][y:y+height, x:x+width]

        #find pixel in CC
        conn_comp_pxl = np.where(sub_bin_filament == idx)
        my_cnnCmp = ConnCompAnalysis(conn_comp_pxl, conn_comp[2][idx])
        if not my_cnnCmp.checkConnCompSize():
            continue
        
        #print(my_cnnCmp.fila_width)
        my_cnnCmp.createGraph()
        my_cnnCmp.pruneGraph()
        my_cnnCmp.createHyperNodes()

        my_cnnCmp.resolveEvenHyperNodes()
        my_cnnCmp.resolveOddNodes()

        if (my_cnnCmp.checkResolvedGraph()):
            #count total number of filaments
            num_filaments += nx.number_connected_components(my_cnnCmp.G)
            num_infected_filaments += my_cnnCmp.countInfected(bin_infection)

        my_cnnCmp.visualizeGraph(sub_img)

        #cv2.imshow('skltn_'+str(idx),my_cnnCmp.skltn*255)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        


    #skeleton_filament = skeletonize(filament,method='lee')
    #cv2.imshow('filament_new',bin_filament*255)
    #cv2.imshow('skeleton',skeleton_filament)

    #overlay segmentation
    res = cv2.addWeighted(pred, 0.1,img, 0.9,0)
    #add text
    res = cv2.putText(res, 'num. living filaments = ' + str(num_filaments), 
                      (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 
    res = cv2.putText(res, 'num. infected filaments = ' + str(num_infected_filaments), 
                      (10,80), cv2.FONT_HERSHEY_SIMPLEX,
                      1, (0,0,0), 2, cv2.LINE_AA) 
    cv2.imwrite("{}.png".format(img_name),res)