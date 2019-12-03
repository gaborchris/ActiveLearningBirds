import tarfile
# import tensorflow as tf
# print( tf.constant('Hello from TensorFlow ' + tf.__version__) )
import pathlib
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import random
import shutil


def download_data(origin='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'):
    data_dir = tf.keras.utils.get_file(origin=origin, fname='bird_photos', untar=True, extract=True)
    return data_dir


class DataHandler:

    def __init__(self, data_path = "C:/Users/Home/.keras/datasets", dataset='CUB_200_2011', target_size=(28, 28), batch_size=32):
        self.data_dir = data_path
        self.dataset_path = os.path.join(pathlib.Path(self.data_dir).parents[0], dataset)
        self.image_data = os.path.join(self.dataset_path, 'images')
        self.target_size = target_size
        self.batch_size = batch_size
        self.flow_gen = self._set_flow_generator()

    def _set_flow_generator(self):
        image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
        image_gen = image_gen.flow_from_directory(self.image_data, target_size=self.target_size,
                                                  batch_size=self.batch_size, class_mode='sparse')
        return image_gen

    def get_flow_generator(self):
        return self.flow_gen

    def get_class_mapping(self):
        pass


### define model
# load pretrained model
# run images through pretrained model without new head
# create new head

## no active learning
# train model on full data
# train model on splits with weighted gradients

### show confusion matrix
# show per class precision, loss, accuracy,
# show confusion matrix

## entropy active learning

## gradient length active learning



if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.create_new_dataset("downsample")
    # print(builder.get_base_class_counts())
    # builder.random_downsample(10)

    # dh = DataHandler(batch_size=256)
    # image_generator = dh.get_flow_generator()
    # class_names = dh.get_class_mapping()
    # print(class_names)
    # print(dh.get_path_map())
    # for batch in image_generator:
    #     image, labels = batch
    #     break
    #
    # for i in range(9):
    #     plt.subplot(3, 3, i+1)
    #     plt.imshow(image[i])
    #     plt.xlabel(class_names[labels[i]])
    # plt.tight_layout()
    # plt.show()
    # plt.close()


