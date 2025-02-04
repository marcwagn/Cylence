import json
import glob
import six
import matplotlib.pyplot as plt

from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset

from keras.callbacks import Callback

from . import losses
from .metrics import mean_iou


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          plots_path = './',
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          optimizer_name='adadelta',
          loss_name='categorical_crossentropy',
          do_augment=False,
          augmentation_name="aug_all"):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if loss_name == 'ignore_zero':
            loss_k = losses.masked_categorical_crossentropy
        elif loss_name == 'categorical_focal':
            loss_k = losses.categorical_focal_loss(gamma=2.0, alpha=0.25)
        elif loss_name == 'categorical_crossentropy':
            loss_k = 'categorical_crossentropy'
        else:
            print('loss not defined!')

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=[mean_iou])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    callbacks = [
        CheckpointsCallback(checkpoints_path)
    ]

    if not validate:
        history = model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        history = model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing)

    with open(checkpoints_path+'_iou.csv', 'w') as f:
        if validate:
            f.write("epoch\tmean_iou_train\tmean_iou_val\n")
            for idx, (train_iou, val_iou) in enumerate(zip(history.history['mean_iou'],history.history['val_mean_iou'])):
                f.write("{}\t{}\t{}\n".format(idx,train_iou,val_iou))
        else:
            f.write("epoch\tmean_IoU_train\n")
            for idx, train_iou in enumerate(history.history['mean_iou']):
                f.write("{}\t{}\n".format(idx,train_iou))

    with open(checkpoints_path+'_loss.csv', 'w') as f:
        if validate:
            f.write("epoch\tloss_train\tloss_val\n")
            for idx, (train_loss, val_loss) in enumerate(zip(history.history['loss'],history.history['val_loss'])):
                f.write("{}\t{}\t{}\n".format(idx,train_loss,val_loss))
        else:
            f.write("epoch\tloss_train\n")
            for idx, train_iou in enumerate(history.history['loss']):
                f.write("{}\t{}\n".format(idx,train_loss))

    plt.plot(history.history['mean_iou'])
    if validate:
        plt.plot(history.history['val_mean_iou'])
    plt.title('Model mean IoU')
    plt.ylabel('Mean IoU')
    plt.xlabel('Epoch')
    if validate:
        plt.legend(['train','test'], loc='upper left')
    plt.savefig(checkpoints_path+'_iou.png')
    plt.close()

    plt.plot(history.history['loss'])
    if validate:
        plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if validate:
        plt.legend(['train','test'], loc='upper left')
    plt.savefig(checkpoints_path+'_loss.png')
    plt.close()
