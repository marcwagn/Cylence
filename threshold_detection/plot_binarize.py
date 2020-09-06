import cv2
import matplotlib.pyplot as plt
import numpy as np

from binarize import filterSize
from dataset import *
from utils import surfacePlot 

from scipy import signal
import scipy

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
        contours, _ = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #counts all contours
        arr_number_contours = np.append(arr_number_contours, [len(contours)])

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
    #no feasible smooth factor found
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

    plt.style.use('ggplot')
    plt.plot(arr_number_contours,'.:g',label='normal')
    plt.plot(arr_number_contours_smooth,'.:b',label='smooth')
    plt.axvline(x=optimal_k,label='optimal threshold [k]')
    plt.title("optimal binarization threshold")
    plt.ylabel("number of connected components")
    plt.xlabel("intensity threshold")
    plt.legend()
    plt.savefig("../results/binarize/numContoursTotal.png")

    plt.style.use('ggplot')
    plt.plot(arr_number_contours,'.:g', label='normal')
    plt.plot(arr_number_contours_smooth,'.:b', label='smooth')
    plt.axvline(x=optimal_k, label='optimal threshold [k]')
    plt.title("subplot: optimal binarization threshold")
    plt.ylabel("number of connected components")
    plt.xlabel("intensity threshold")
    plt.xlim((30,200))
    plt.ylim((0,150))
    plt.savefig("../results/binarize/numContoursSubregion.png")

    #create binary image
    _ ,img_binary = cv2.threshold(img_blur,optimal_k,1,cv2.THRESH_BINARY)

    return img_binary


if __name__ == "__main__":
    #import image as color image
    imgData = dataLoader('../data/',1)
    for image in imgData:

        id, bright, blue, red = image

        #select color specific channel
        red_gray = red[:,:,2]
        blue_gray = blue[:,:,0]
        surfacePlot(blue_gray,"../results/surface/blueGray.png")

        #create binary masks
        #red_binary = binarizeImg(red_gray)
        #blue_binary = binarizeImg(blue_gray)

        #red_binary = imgWizard.filterSize(red_binary,625)
        #blue_binary = imgWizard.filterSize(blue_binary,625)

        #cv2.imwrite("../results/binarize/blueBinary.png",blue_binary*255)
        #cv2.imwrite("../results/binarize/redBinary.png",red_binary*255)

        #img_size = 400
        #window = cv2.namedWindow('img'+id, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('img'+id, img_size,img_size)
        #cv2.imshow('img'+id,blue_gray)


        #window = cv2.namedWindow('img2'+id, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('img2'+id, img_size,img_size)
        #cv2.imshow('img2'+id,blue_binary*255)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        exit()
