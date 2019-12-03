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
import json

DATA_ROOT = "C:/Users/Home/.keras/datasets/"


def get_dir_mapping(path):
    dirs = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
    num_classes = len(dirs)
    dir_map = {}
    for i in range(0, num_classes):
        images_path = os.path.join(path, dirs[i])
        dir_map[i] = [os.path.join(dirs[i], x) for x in os.listdir(images_path)]
    return dir_map


def write_json_dict(path, fname, dict_map):
    json_dict = json.dumps(dict_map)
    class_mapping_path = os.path.join(path, fname)
    with open(class_mapping_path, 'w') as f:
        f.write(json_dict)


def get_class_counts(dir_map):
    counts = {}
    for k, v in dir_map.items():
        counts[int(k)] = len(v)
    return counts


class DatasetBuilder:

    def __init__(self, dataset_dir="C:/Users/Home/.keras/datasets/", base_dir="CUB_200_2011"):
        self.dataset_dir = dataset_dir
        self.base_dir = os.path.join(self.dataset_dir, base_dir)
        self.image_dir = os.path.join(self.base_dir, 'images')

    def get_base_class_mapping(self):
        class_path = os.path.join(self.base_dir, 'classes.txt')
        with open(class_path) as file:
            classes = dict((line.strip().split(' ') for line in file))
        classes = {(int(k)-1):v for k,v in classes.items()}
        return classes

    def random_downsample(self, num_classes, ratio):
        dir_map = get_dir_mapping(self.image_dir)
        new_set = {}
        targets = random.sample(list(dir_map), num_classes)
        for k, v in dir_map.items():
            if k in targets:
                count = len(v)
                sample_size = math.ceil(ratio*count)
                v = random.sample(v, sample_size)
            new_set[k] = v
        return new_set

    def create_new_dataset(self, target, num_downsample=200, ratio=0.05):
        new_data_path = os.path.join(self.dataset_dir, target)
        new_image_path = os.path.join(new_data_path, 'images')
        new_dataset = self.random_downsample(num_downsample, ratio)
        for k, v in new_dataset.items():
            print("Creating dir: ", k)
            for path in v:
                source = os.path.join(self.image_dir, path)
                dest = os.path.join(new_image_path, path)
                print("Copying ", source, end=' ')
                print("to ", dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copyfile(source, dest)

        write_json_dict(new_data_path, 'class_mapping.json', self.get_base_class_mapping())
        write_json_dict(new_data_path, 'class_counts.json', get_class_counts(get_dir_mapping(new_image_path)))


if __name__ == "__main__":
    builder = DatasetBuilder()
    # dir_map = builder.random_downsample(140, 0.1)
    builder.create_new_dataset('one_percent',200, 0.01)

