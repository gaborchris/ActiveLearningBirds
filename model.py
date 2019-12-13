import tensorflow as tf
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def create_model(verbose=False):
    reg = 0
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.5, input_shape=(1280,)),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(reg))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.top_k_categorical_accuracy])
    if verbose:
        model.summary()
    return model


def get_class_weight(y):
    sparse = np.argmax(y, axis=1)
    unique, counts = np.unique(sparse, return_counts=True)
    counts = np.column_stack((unique, counts))
    max_freq = np.max(counts[:, 1])
    weights = dict((counts[i][0], max_freq / float(counts[i][1])) for i in range(len(counts)))
    return weights


class Model:

    def __init__(self, verbose=True):
        self.v = verbose
        self.model = create_model(self.v)
        self.class_weights = {}

    def set_class_weights(self, y):
        self.class_weights = get_class_weight(y)

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1, test_val=False, verbose=True):
        if test_val:
            self.model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), class_weight=self.class_weights,
                           epochs=epochs, verbose=verbose)
        else:
            self.model.fit(x=x_train, y=y_train, class_weight=self.class_weights,
                           epochs=epochs, verbose=verbose)

    def predict(self, x, verbose=True):
        return self.model.predict(x, verbose=verbose)

    def save(self, path, save_format='tf'):
        self.model.save(path) #, save_format=save_format)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
