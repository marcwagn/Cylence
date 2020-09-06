import sys
sys.path.insert(1, '/home/marc/Dropbox/WiSe1920/MA_thesis/03_KERAS_detection/image-segmentation-keras')

import cv2
import numpy as np

from keras_segmentation.data_utils.data_loader import image_segmentation_generator
from keras_segmentation.predict import visualize_segmentation


train_images = "../dataset/test_aug/images_train"
train_annotations = "../dataset/test_aug/annotations_train/"
batch_size = 2
n_classes = 3
input_height = 480
input_width = 640
output_height = 240
output_width = 320
do_augment = True
augmentation_name = 'test'

train_gen = image_segmentation_generator(
    train_images, train_annotations,  batch_size,  n_classes,
    input_height, input_width, output_height, output_width,
    do_augment=do_augment, augmentation_name=augmentation_name)

for i in range(1):
    img_set , seg_set = next(train_gen)
    seg = seg_set[0]
    seg = seg.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    #print(seg.shape)
    seg = visualize_segmentation(seg,n_classes=3,colors=[(0,255,0),(0,0,255),(255,0,0)])
    #pred = cv2.resize(pred,(640,480), interpolation = cv2.INTER_NEAREST)
    img = img_set[0]
    img = img[:,:,::-1]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    #print(np.max(img), np.min(img))
    cv2.imwrite('../../thesis/images/aug/MotionBlur_-45.png',img.astype(np.uint8))
    #cv2.imshow('segmentation',seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
