import sys
import glob
import cv2
import os
import numpy as np

from .connCompAnalysis import ConnCompAnalysis

from skimage.morphology import skeletonize
from scipy import signal

import networkx as nx

import matplotlib.pyplot as plt

def quantify(img, pred):
    """
    args:
        pred: colored image prediction
    returns:
        num_filaments: number of filaments 
        num_infected_filaments: number infected
    """
    #create binary masks
    bin_infection = pred[:,:,2] // 255
    bin_filament = pred[:,:,1] // 255

    #closing binary filament
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bin_filament = cv2.morphologyEx(bin_filament, cv2.MORPH_CLOSE, kernel)

    #find connected components
    conn_comp = cv2.connectedComponentsWithStats(bin_filament , cv2.CV_32S, connectivity=8)

    #result parameter
    num_filaments = 0
    num_infected_filaments = 0

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
        
        my_cnnCmp.createGraph()
        my_cnnCmp.pruneGraph()
        my_cnnCmp.createHyperNodes()

        my_cnnCmp.resolveEvenHyperNodes()
        my_cnnCmp.resolveOddNodes()
        
        my_cnnCmp.removeUnresolved()

        if (my_cnnCmp.checkResolvedGraph()):
            #count total number of filaments
            newFila, newInf = my_cnnCmp.quantifyParameter(bin_infection)
            num_filaments += newFila
            num_infected_filaments += newInf
        
        my_cnnCmp.visualizeGraph(sub_img)
    
    return num_filaments, num_infected_filaments


if __name__ == "__main__":
    path = "/home/marc/Dropbox/WiSe1920/MA_thesis/database/test"
    img_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob("{}/predictions/*.png".format(path))]
    print(img_name_arr)
    for img_name in img_name_arr:

        #load data
        img = cv2.imread("{}/images/{}.png".format(path,img_name), )
        pred = cv2.imread("{}/predictions/{}.png".format(path,img_name), )

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