"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.09.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   CNN test predict
########################################################
"""

import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

from image_segmentation_keras.keras_segmentation.predict import model_from_checkpoint_path
from image_segmentation_keras.keras_segmentation.predict import predict
from image_segmentation_keras.keras_segmentation.predict import visualize_segmentation

def predictTiled(img, model):
    """
    Args:
        img: rgb image (cyanos, cytrid, background)
        model: image segmentation CNN
    Returns: (pred, pred_img)
        pred: labeled pixel (cyanos = 0, cytrid = 1, background = 2)
        pred_img: colored pred 
    """
    h_img, w_img, _ = img.shape

    pred = np.zeros((h_img,w_img),dtype='uint8')
    pred_img = np.zeros_like(img)

    if (h_img < 480) or (w_img < 640):
        raise ValueError("Image size too small for model!")

    w_pos = list(range(0, (w_img-640)+1, 640))
    h_pos = list(range(0,(h_img-480)+1,480))

    if (w_img % 640 != 0):
        w_pos.append(w_img-640)
    if (h_img % 480 != 0):
        h_pos.append(h_img-480)
    
    for w_iter in tqdm(w_pos):
        for h_iter in tqdm(h_pos):
            sub_img = img[h_iter:h_iter+480,w_iter:w_iter+640,:]

            sub_pred = predict(model,inp=sub_img)
            sub_pred = cv2.resize(sub_pred,(640,480), interpolation = cv2.INTER_NEAREST)
            pred[h_iter:h_iter+480,w_iter:w_iter+640] = sub_pred

            sub_pred_img = visualize_segmentation(sub_pred,n_classes=3,colors=[(0,255,0),(0,0,255),(255,0,0)])
            pred_img[h_iter:h_iter+480,w_iter:w_iter+640,:] = sub_pred_img

    return pred, pred_img


if __name__ == "__main__":
    #path = "../../database/02_UnetSingle/images/"
    #path = "../dataset/raw/images/"
    path = "../dataset/tiled/raw/test2/"
    ftype = ".tif"

    model = model_from_checkpoint_path("../model/vgg_unet_cross_aug_ep100_51_100/vgg_unet_1")

    img_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(path + "*" + ftype)]

    for img_name in img_name_arr:
        img = cv2.imread(path+img_name+ftype)

        pred, pred_img = predictTiled(img,model)

        #overlay = cv2.addWeighted(pred_img, 0.1,img, 0.9,0,dtype = cv2.CV_32F)

        cv2.imwrite("../dataset/tiled/raw/predictions/"+img_name+".png", pred_img)

        #cv2.imshow('image',img)
        #cv2.imshow('prediction',pred_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

