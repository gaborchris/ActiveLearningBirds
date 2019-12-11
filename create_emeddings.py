import tensorflow as tf
import tensorflow_hub as hub
import ipykernel
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import time

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


IMAGE_SHAPE = (224, 224)


def load_flowers():
    data_root = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)
    return data_root

def load_birds(dataset='full_data'):
    data_root = os.path.join("/home/christian/.keras/datasets/", dataset)
    data_root = os.path.join(data_root, "images")
    return data_root


def get_mobilenet_feature_extractor():
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=IMAGE_SHAPE+(3,))
    return feature_extractor_layer


# TODO implement preprocess rotation, flip, shear, color_shift
def create_preprocessed_embeddings():
    pass


def get_embeddings(data_root, base, target_name):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE, batch_size=1, shuffle=False)

    # feature_batch = base(image_batch)
    base.trainable = False
    model = tf.keras.Sequential([
        base,
    ])
    EMB_DIM = model.output_shape[1]
    LBL_DIM = image_data.num_classes
    print("Creating embedding with output shape {} belonging to {} classes".format(EMB_DIM, LBL_DIM))
    time.sleep(2)

    num_samples = image_data.samples

    x_embed = np.zeros((num_samples, EMB_DIM))
    labels = np.zeros((num_samples, LBL_DIM))
    source_images = []
    for i in range(num_samples):
        idx = image_data.batch_index
        image, label = next(image_data)
        file = image_data.filenames[idx]
        print(idx, file)
        source_images.append(file)
        x_embed[idx] = model(image).numpy()
        labels[idx] = label
    np.savez('./embeddings/'+target_name, x_embed, labels, source_images)


def create_train_test(name, train_split):
    path = "./embeddings/"+name+".npz"
    with np.load(path) as data:
        data_x = data['arr_0']
        data_y = data['arr_1']
        names = data['arr_2']
    p = np.random.permutation(len(data_x))
    x_shuffle, y_shuffle, sources_shuffle = data_x[p], data_y[p], names[p]
    train_split = int(x_shuffle.shape[0]*train_split)
    x_train, y_train, sources_train = x_shuffle[:train_split], y_shuffle[:train_split], sources_shuffle[:train_split]
    x_test, y_test, sources_test = x_shuffle[train_split:], y_shuffle[train_split:], sources_shuffle[train_split:]
    np.savez('./embeddings/'+name+'_train', x_train, y_train, sources_train)
    np.savez('./embeddings/'+name+'_test', x_test, y_test, sources_test)



def save_embed_to_seperate_files(data_root, base):
    pass
    # for i in range(image_data.samples):
    #     idx = image_data.batch_index
    #     image, label = next(image_data)
    #     print(label)
    #
    #     get directory to save
    #     file = image_data.filenames[idx]
    #     path = os.path.abspath("./embeddings")
    #     class_label = os.path.dirname(file)
    #     save_dir = os.path.join(path, class_label)
    #     embed_file_path = os.path.join(path, os.path.splitext(image_data.filenames[idx])[0])
    #
    #     create directory for class
    #     if not os.path.exists(save_dir):
    #         print('true')
    #         os.makedirs(save_dir)
    #
    #     print("saving to", embed_file_path)
    #     np.save(embed_file_path, model(image).numpy())


if __name__ == "__main__":
    # get_embeddings(load_flowers(), get_mobilenet_feature_extractor())
    # get_embeddings(load_birds(), get_mobilenet_feature_extractor(), 'full_bird_data')
    # get_embeddings(load_birds(dataset='top_half_tenth'), get_mobilenet_feature_extractor(), 'half_tenth')
    # get_embeddings(load_birds(dataset='bird_train_full'), get_mobilenet_feature_extractor(), 'train_full')
    # get_embeddings(load_birds(dataset='bird_train_sampled'), get_mobilenet_feature_extractor(), 'train_sampled')
    get_embeddings(load_birds(dataset='bird_test_data'), get_mobilenet_feature_extractor(), 'test')
