"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.07.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   threshold binarization
########################################################
"""

import numpy as np
import cv2

import scipy.ndimage
from scipy import signal

def binarizeImg(img):
    """
    Args:
        img: grayscale image
    Returns:
        binary image (ndarray numpy) based on optimal threshold
    """
    #remove gaussian noise from image
    img_blur = cv2.GaussianBlur(img,(11,11),0)

    # find best threshold for binarizing the image
    arr_number_contours = np.array([], dtype=int)
    for k in range(1,255,1):
        #binaries image based on threshold k
        _ ,img_binary = cv2.threshold(img_blur,k,255,cv2.THRESH_BINARY)
        #find all contours
        conn_comp = cv2.connectedComponents(img_binary , cv2.CV_32S, connectivity=8)
        #counts all contours
        arr_number_contours = np.append(arr_number_contours, [conn_comp[0]])

    #smooth number of contours list to find maxima
    smoothFactor = 10
    while (smoothFactor > 1):
        arr_number_contours_smooth = scipy.ndimage.gaussian_filter(arr_number_contours, smoothFactor)
        #find all local maxima in number of contours in smoothed dataset
        #(local maxima => intensity lvl of a new object on the image)
        lst_maximal_number_contours = scipy.signal.find_peaks(arr_number_contours_smooth)[0]
        if len(lst_maximal_number_contours) < 2:
            smoothFactor -= 1
        else:
            break
    #no feasible smooth factor could be found
    if len(lst_maximal_number_contours) < 2:
        raise IndexError("List of intensity peaks contains <2 elements!")

    #find all minima between maxima in non smoothed numContours List
    lst_minimal_number_contours = []
    for i in range(0,len(lst_maximal_number_contours)-1):
        lower_bound = lst_maximal_number_contours[i]
        upper_bound = lst_maximal_number_contours[i+1]
        lst_minimal_number_contours.append((lower_bound,
                                            upper_bound, 
                                            arr_number_contours[lower_bound:upper_bound].min())
                                        )

    #find largest minimum and all occourences in the given bounds
    lower_bound, upper_bound, number_contours = max(lst_minimal_number_contours, key = lambda t: t[2])
    lst_optimal_k = np.where(arr_number_contours[lower_bound:upper_bound] == number_contours)+lower_bound+1

    #optimal threshold k
    optimal_k = lst_optimal_k[0].min()

    #create binary image
    _ ,img_binary = cv2.threshold(img_blur,optimal_k,1,cv2.THRESH_BINARY)

    return img_binary

def filterSize(img, min_pixel):
    """
    Args:
        img: binary image
        min_pixel: minimal size of contour
    Returns:
        removes all connected components < min_pixel in img
    """
    conn_comp = cv2.connectedComponentsWithStats(img, cv2.CV_32S, connectivity=8)

    #iterate over all connected components
    for idx in range(1,conn_comp[0]):
        comp_pxl = np.where(conn_comp[1] == idx)
        num_pxl = len(comp_pxl[0])
        if num_pxl < min_pixel:
            img[comp_pxl] = 0