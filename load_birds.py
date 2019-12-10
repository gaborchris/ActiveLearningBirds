import tensorflow as tf
print( tf.constant('Hello from TensorFlow ' + tf.__version__) )
import os
import matplotlib.pyplot as plt
import json


def download_data(origin='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'):
    data_dir = tf.keras.utils.get_file(origin=origin, fname='bird_photos', untar=True, extract=True)
    return data_dir


def get_flow_generator(path, dataset_name, target_size, batch_size, mode='sparse'):
    dataset_path = os.path.join(path, dataset_name)
    image_data = os.path.join(dataset_path, 'images')
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
    image_gen = image_gen.flow_from_directory(image_data, target_size=target_size, batch_size=batch_size, class_mode=mode)
    return image_gen


def parse_json_classes(path, dataset, fname):
    dataset_path = os.path.join(path, dataset)
    with open(os.path.join(dataset_path, fname)) as f:
        parsed = json.load(f)
    mapping = {}
    for k, v in parsed.items():
        mapping[int(k)] = v
    return mapping


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
    # download_data()
    path = "/home/christian/.keras/datasets"
    dataset_name = 'one_percent'

    im_gen = get_flow_generator(path=path, dataset_name=dataset_name, batch_size=32, target_size=(224, 224))

    class_names = parse_json_classes(path, dataset_name, 'class_mapping.json')
    class_counts = parse_json_classes(path, dataset_name, 'class_counts.json')
    print(class_counts)

    # print(class_names)
    # print(dh.get_path_map())
    for batch in im_gen:
        image, labels = batch
        break

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(image[i])
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    plt.show()
    plt.close()


