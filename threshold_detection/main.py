import cv2
import numpy as np

from dataset import *
from binarize import binarizeImg, filterSize

if __name__ == "__main__":
    #import image as color image
    imgData = dataLoader('../data/',1)
    for image in imgData:

        id, bright, infection, filament = image

        if id != "1022":
            continue

        #cv2.imwrite("../results/binarize/{}_infection.png".format(id),infection)
        #cv2.imwrite("../results/binarize/{}_filament.png".format(id),filament)

        #select color specific channel
        filament_gray = filament[:,:,2]
        infection_gray = infection[:,:,0]

        #create binary masks
        bin_filament = binarizeImg(filament_gray)
        bin_infection = binarizeImg(infection_gray)

        #create binary mask OTSU
        blur = cv2.GaussianBlur(filament_gray,(11,11),0)
        _,bin_filament_otsu = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #cv2.imwrite("../results/binarize/{}_binary_infection.png".format(id),bin_infection*255)
        #cv2.imwrite("../results/binarize/{}_binary_filament.png".format(id),bin_filament*255)
        #cv2.imwrite("../results/binarize/{}_binary_filament_otsu.png".format(id),bin_filament_otsu*255)

        #closing (binary filament)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        bin_filament = cv2.morphologyEx(bin_filament, cv2.MORPH_CLOSE, kernel)

        #size filter binary masks
        #filterSize(bin_filament,625)
        #filterSize(bin_infection,625)

        #cv2.imwrite("../results/binarize/{}_filter_infection.png".format(id),bin_infection*255)
        #cv2.imwrite("../results/binarize/{}_filter_filament.png".format(id),bin_filament*255)

        #test
        h, w = filament_gray.shape
        sep = np.full((h,10,3),255)
        color_otsu = np.dstack((bin_filament_otsu,bin_filament_otsu,bin_filament_otsu))* 255
        color_filament = np.dstack((bin_filament,bin_filament,bin_filament))* 255

        comb_color = np.concatenate((filament,sep, color_filament),axis=1)

        cv2.imwrite("../results/binarize_selected/{}_degrading_color.png".format(id),comb_color)
