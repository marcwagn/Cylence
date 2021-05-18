"""
########################################################
o       o
 \_____/ 
 /=O=O=\     _______ 
/   ^   \   /\\\\\\\\
\ \___/ /  /\   ___  \\\ |author:  Marc-Christoph Wagner
 \_ V _/  /\   /\\\\  \\\|date:    01.09.2020
   \  \__/\   /\ @_/  ///|project: master thesis
    \____\____\______/// |topic:   CNN test train
########################################################
"""

from image_segmentation_keras.keras_segmentation.models.unet import vgg_unet
from image_segmentation_keras.keras_segmentation.models.fcn import fcn_8_vgg
from image_segmentation_keras.keras_segmentation.models.segnet import vgg_segnet
from image_segmentation_keras.keras_segmentation.models.unet import unet_mini
from image_segmentation_keras.keras_segmentation.models.unet import resnet50_unet


vgg_unet_model = vgg_unet(n_classes=3, input_height=480, input_width=640)
mini_unet_model = unet_mini(n_classes=3, input_height=480, input_width=640)
resnet_unet_model = resnet50_unet(n_classes=3, input_height=480, input_width=640)
fcn8_model = fcn_8_vgg(n_classes=3, input_height=480, input_width=640)
segnet_model = vgg_segnet(n_classes=3, input_height=480, input_width=640)

print('resnet_unet')
resnet_unet_model.summary()
print('mini_unet')
mini_unet_model.summary()

#model.train(
#    train_images = "../dataset/filter/images_train",
#    train_annotations = "../dataset/filter/annotations_train/",
#    checkpoints_path = "../tmp/vgg_unet_1",
#    validate = True,
#    val_images = "../dataset/filter/images_test",
#    val_annotations = "../dataset/filter/annotations_test/",
#    val_steps_per_epoch = 5,
#    epochs=2, 
#    steps_per_epoch=5,
#    loss_name='categorical_focal',
##    loss_name='categorical_crossentropy',
#    verify_dataset=False,
#    do_augment = False,
#    augmentation_name = 'simple'
#)