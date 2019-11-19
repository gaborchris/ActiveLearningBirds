import tarfile
import tensorflow as tf
import pathlib

data_dir = tf.keras.utils.get_file(origin='http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
                                   fname='bird_photos', untar=True)
data_dir = pathlib.Path(data_dir)




def unzip_data(path):
    with tarfile.open(path, "r:gz") as so:
        so.extractall(path="./data")
    print('hi')


if __name__ == "__main__":
    # unzip_data("./data/CUB_200_2011.tgz")
    pass
