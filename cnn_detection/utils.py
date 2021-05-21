"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.09.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   CNN utils
########################################################
"""

import cv2
import glob
import os
import shutil
from tqdm import tqdm
import numpy as np

from random import shuffle

from .image_segmentation_keras.keras_segmentation.predict import predict
from .image_segmentation_keras.keras_segmentation.predict import visualize_segmentation

def predictTiled(img, model):
    """
    Args:
        img: rgb image (cyanos, cytrid, background)
        model: image segmentation CNN
    Returns: (pred, pred_img)
        pred_img: colored pred 
    """
    h_img, w_img, _ = img.shape

    pred_img = np.zeros_like(img)

    if (h_img < 480) or (w_img < 640):
        raise ValueError("Image size too small for model!")

    w_pos = list(range(0, (w_img-640)+1, 640))
    h_pos = list(range(0,(h_img-480)+1,480))

    if (w_img % 640 != 0):
        w_pos.append(w_img-640)
    if (h_img % 480 != 0):
        h_pos.append(h_img-480)
    
    for w_iter in w_pos:
        for h_iter in h_pos:
            sub_img = img[h_iter:h_iter+480,w_iter:w_iter+640,:]

            sub_pred = predict(model,inp=sub_img)
            sub_pred = cv2.resize(sub_pred,(640,480), interpolation = cv2.INTER_NEAREST)

            sub_pred_img = visualize_segmentation(sub_pred,n_classes=3,colors=[(0,255,0),(0,0,255),(255,0,0)])
            pred_img[h_iter:h_iter+480,w_iter:w_iter+640,:] = sub_pred_img

    return pred_img

def createSubimages(img_dir, anno_dir, out_dir, relabel=False):
    """
    Args:
        img_dir: image directory [960x1280, height x width]
        anno_dir: annotations directory [960x1280, height x width]
        out_dir: output directory
    Returns:
        subimages [480x640, height x width]
        subanno [240x480,height x width]
        filter: filtered all imgs with just background
        prepped: no filtering step included
    """
    #load image filenames
    img_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(img_dir + str("/*.png"))]

    #clear output folder
    dir_lst = [ out_dir + "/background/images/",
            out_dir + "/background/annotations/",
            out_dir + "/filament/images/",
            out_dir + "/filament/annotations/",
            out_dir + "/infection/images/",
            out_dir + "/infection/annotations/"]

    for dir in dir_lst:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)
    
    #create subimages
    for img_name in tqdm(img_name_arr):
        img = cv2.imread(img_dir+img_name+".png",1)
        ann = cv2.imread(anno_dir+img_name+".png",1)
        h_img, w_img, _ = img.shape
        
        #correct labeling
        if relabel == True:
            ann[ann == 1] = 0
            ann[ann == 2] = 1
            ann[ann == 3] = 2

        idx = 1

        for w_iter in range(0, (w_img-640)+1, 640):
            for h_iter in range(0,(h_img-480)+1,480):
                #create subimages
                sub_img = img[h_iter:h_iter+480,w_iter:w_iter+640,:]
                sub_ann = ann[h_iter:h_iter+480,w_iter:w_iter+640,:]

                #write background subimages/annotations
                if not (0 in sub_ann) and not (1 in sub_ann):
                    cv2.imwrite(out_dir + "/background/images/"+img_name+"_"+str(idx)+".png", sub_img)
                    cv2.imwrite(out_dir + "/background/annotations/"+img_name+"_"+str(idx)+".png", sub_ann)

                #write filament subimages/annotations
                if (0 in sub_ann) and not (1 in sub_ann):
                    cv2.imwrite(out_dir + "/filament/images/"+img_name+"_"+str(idx)+".png", sub_img)
                    cv2.imwrite(out_dir + "/filament/annotations/"+img_name+"_"+str(idx)+".png", sub_ann)

                #write filtered subimages/annotations
                if (0 in sub_ann) and (1 in sub_ann):
                    cv2.imwrite(out_dir + "/infection/images/"+img_name+"_"+str(idx)+".png", sub_img)
                    cv2.imwrite(out_dir + "/infection/annotations/"+img_name+"_"+str(idx)+".png", sub_ann)
                
                idx += 1

def splitTestTrain(img_dir, anno_dir, out_dir, perc):
    """
    Args:
        img_dir: image directory
        anno_dir: annotations directory
        out_dir: output directory
        perc: percentage test set 
    Returns:
        test and training set in output folder
    """
    #load image filnames
    img_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(img_dir + str("/*.png"))]

    #randomize train and test set
    shuffle(img_arr)

    assert (perc <= 1 and perc >= 0), "perc value must be 0 <= x <= 1"
    split = int(len(img_arr) * perc)

    test_img_arr = img_arr[:split]
    train_img_arr = img_arr[split:]

    #clear output folder
    dir_lst = [ out_dir + "/images_test/",
                out_dir + "/annotations_test/",
                out_dir + "/images_train/",
                out_dir + "/annotations_train/"]

    for dir in dir_lst:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            shutil.rmtree(dir)
            os.makedirs(dir)

    #generate test data
    for test_img in test_img_arr:
        tmp_img = cv2.imread(img_dir+test_img+".png",1)
        tmp_ann = cv2.imread(anno_dir+test_img+".png",1)

        cv2.imwrite(out_dir + "images_test/" + test_img + ".png", tmp_img)
        cv2.imwrite(out_dir + "annotations_test/" + test_img + ".png", tmp_ann)


    #generate train data
    for train_img in train_img_arr:
        tmp_img = cv2.imread(img_dir+train_img+".png",1)
        tmp_ann = cv2.imread(anno_dir+train_img+".png",1)

        cv2.imwrite(out_dir + "images_train/"+train_img+".png", tmp_img)
        cv2.imwrite(out_dir + "annotations_train/"+train_img+".png", tmp_ann)

def correctLabel(ann_dir, out_dir):
    """
    -helper function-
    Returns: correct labels
    """
    ann_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob("{}/*.png".format(ann_dir))]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    for ann_name in tqdm(ann_name_arr):
        ann = cv2.imread("{}{}.png".format(ann_dir,ann_name),1)

        ann[ann == 1] = 0
        ann[ann == 2] = 1
        ann[ann == 3] = 2

        cv2.imwrite("{}{}.png".format(out_dir,ann_name), ann)

def labelToRGB(ann_dir, out_dir):
    """
    Args:
        labeled annotation
            back([2,..]), inf([1,..]),fila([0,..]) 
    Returns:
        colored annotation:
            back(blue), inf(red), fila(green)
    """
    #load annotation filenames
    ann_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob("{}/*.png".format(ann_dir))]

    #clear output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    colors = [(0,255,0),(0,0,255),(255,0,0)]

    for ann_name in tqdm(ann_name_arr):
        ann = cv2.imread(ann_dir+ann_name+".png",1)

        for c in range(3):
            ann[np.where(np.all(ann == [c,c,c], axis=-1))] = colors[c]
        
        cv2.imwrite("{}/{}.png".format(out_dir,ann_name), ann)

def rgbToLabel(ann_dir, out_dir):
    """
    Args:
        colored annotation:
            back(blue), inf(red), fila(green)
    Returns:
        labeled annotation
            back([2,..]), inf([1,..]),fila([0,..]) 
    """
    #load annotation filenames
    ann_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob("{}/*.png".format(ann_dir))]

    #clear output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    colors = [[0,255,0],[0,0,255],[255,0,0]]

    for ann_name in tqdm(ann_name_arr):
        print(ann_dir+ann_name+".png")
        ann_color = cv2.imread(ann_dir+ann_name+".png",1)
        ann = np.zeros(ann_color.shape, np.uint8)
        ann.fill(255)

        for c in range(3):
            ann[np.where(np.all(ann_color == colors[c], axis=-1))] = [c,c,c]
        
        if 255 in ann:
            print("Error in file: {}".format(ann_name))
        else:
            cv2.imwrite("{}/{}.png".format(out_dir,ann_name), ann)

def onehotToRGB(onehot_mask):
    """
    Args:
        segmentation: one hot encoded mask with 3 classes
    Returns:
        seg image:
            background(2 = blue)
            infection(1 = red)
            filaments(0 = green)
    """
    colors = [(0,255,0),(0,0,255),(255,0,0)]
    h, w, _ = onehot_mask.shape
    seg_img = np.zeros((h,w,3)).astype('uint8')

    for c in range(3):
        seg_img[:, :, 0] += ((onehot_mask[:,:,c] == 1) *
                                (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((onehot_mask[:,:,c] == 1) *
                                (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((onehot_mask[:,:,c] == 1) *
                                (colors[c][2])).astype('uint8')

    return seg_img

def visulizeSample(img, true_mask, pred, outfile):
    """
    Args:
        img: original image (rgb)
        onehot_mask: one hot encoded segmentation mask
        outfile: path for output file
        pred: 
    Returns:
        orginal + segmentation + prediction + overlay
    """
    #parameter
    h, w, _ = img.shape

    #overlay img and mask
    overlay = cv2.addWeighted(pred, 0.1,img, 0.9,0,dtype = cv2.CV_32F)

    #label images
    text = "image"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text,font,1,2)[0]
    textX = (w - textsize[0]) // 2
    textY = 30
    cv2.putText(img, text, 
                (textX, textY),font,
                1, (0,0,0), 2, cv2.LINE_AA)

    text = "segmentation"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text,font,1,2)[0]
    textX = (w - textsize[0]) // 2
    textY = 30
    cv2.putText(true_mask, text, 
                (textX, textY), font,
                1, (0,0,0), 2, cv2.LINE_AA)

    text = "prediction"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text,font,1,2)[0]
    textX = (w - textsize[0]) // 2
    textY = 30
    cv2.putText(pred, text, 
                (textX, textY), font,
                1, (0,0,0), 2, cv2.LINE_AA)

    text = "overlay"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text,font,1,2)[0]
    textX = (w - textsize[0]) // 2
    textY = 30
    cv2.putText(overlay, text, 
                (textX, textY), font,
                1, (0,0,0), 2, cv2.LINE_AA)

    #create output image
    sep_hori = np.full((h, 5, 3), 255)
    sep_vert = np.full((5, 2*w + 5, 3), 255)

    row1 = np.concatenate((img, sep_hori, overlay), axis=1)
    row2 = np.concatenate((true_mask, sep_hori, pred), axis=1)
    res = np.concatenate((row1, sep_vert, row2), axis=0)

    cv2.imwrite(outfile, res)
    return res

def clearAnnotations(img_dir, anno_dir):
    img_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(img_dir + str("/*.png"))]
    anno_name_arr = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(anno_dir + str("/*.png"))]
    
    for anno_name in tqdm(anno_name_arr):
        if not (anno_name in img_name_arr):
            os.remove(anno_dir+anno_name+".png")
        #anno = cv2.imread(anno_dir+img_name+".png",1)
        #cv2.imwrite(out_dir +img_name+".png", anno)

if __name__ == "__main__":
    #clearAnnotations("../../data/10x_nikon_aldehyd/filament/images","../../data/10x_nikon_aldehyd/filament/annotations/")
    #createSubimages("../../data/10x_nikon_aldehyd/raw/images/", 
    #                "../../data/10x_nikon_aldehyd/raw/annotations/", 
    #                "../../data/10x_nikon_aldehyd/", relabel=False)
    #splitTestTrain("../../data/final/images/","../../data/final/annotations/","../../data/final/", 0.2)
    #labelToRGB("../dataset/small/filter/annotations/", "../dataset/small/filter/annotaions_color/")
    #rgbToLabel("../../data/10x_nikon_aldehyd/raw/annotations_color/","../../data/10x_nikon_aldehyd/raw/annotations/")
    pass