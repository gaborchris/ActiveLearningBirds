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

DATA_ROOT = "/home/christian/.keras/datasets/"


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

    def __init__(self, dataset_dir=DATA_ROOT, base_dir="CUB_200_2011"):
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

    def random_downsample_range(self, start_idx, ratio):
        dir_map = get_dir_mapping(self.image_dir)
        new_set = {}
        targets = list(dir_map)[start_idx:]
        for k, v in dir_map.items():
            if k in targets:
                count = len(v)
                sample_size = math.ceil(ratio*count)
                v = random.sample(v, sample_size)
            new_set[k] = v
        return new_set

    def train_test_split(self, ratio=0.2):
        dir_map = get_dir_mapping(self.image_dir)
        train_set = {}
        test_set = {}
        for k, v in dir_map.items():
            count = len(v)
            split = math.ceil(ratio*count)
            random.shuffle(v)
            test = v[:split]
            train = v[split:]
            train_set[k] = train
            test_set[k] = test
        return train_set, test_set

    def random_downsample_map_range(self, mapping, start_idx, ratio):
        new_set= {}
        targets = list(mapping)[start_idx:]
        for k, v in mapping.items():
            if k in targets:
                count = len(v)
                split = math.ceil(ratio*count)
                v = v[:split]
            new_set[k]= v
        return new_set


    def create_new_dataset(self, target, mapping):
        new_data_path = os.path.join(self.dataset_dir, target)
        new_image_path = os.path.join(new_data_path, 'images')
        for k, v in mapping.items():
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
    train_map, test_map = builder.train_test_split(0.2)
    down_sample = builder.random_downsample_map_range(train_map, 100, 0.1)

    # Test dataset
    builder.create_new_dataset('bird_test_data', test_map)
    # Train dataset full
    builder.create_new_dataset('bird_train_full', train_map)
    # Train dataset downsampled
    builder.create_new_dataset('bird_train_sampled', down_sample)

    # builder.create_new_dataset('top_half_tenth', 'idx', 100, 0.1)
    # dir_map = builder.random_downsample(140, 0.1)
    # builder.create_new_dataset('one_percent',200, 0.01)
    # builder.create_new_dataset('full_data',200, 1.0)

