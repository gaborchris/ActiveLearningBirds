import tensorflow as tf
import tensorflow_hub as hub
import ipykernel
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def load_train_embed(path):
    with np.load(path) as data:
        train_x = data['arr_0']
        train_y = data['arr_1']
        names = data['arr_2']
    return train_x, train_y, names


def get_metrics(y_true, y_pred, idx=None, verbose=True):
    confusion = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)
    accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
    correct = np.diag(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    recall = np.zeros(shape=correct.shape)
    precision = np.zeros(shape=correct.shape)
    for i, c in enumerate(correct):
        if row_sums[i] != 0:
            recall[i] = c / row_sums[i]
        if col_sums[i] != 0:
            precision[i] = c / col_sums[i]
    if idx is not None:
        recall = recall[idx[0]: idx[1]]
        precision = precision[idx[0]: idx[1]]
    if verbose:
        print("Accuracy:", accuracy)
        print("Average Recall:", np.average(recall))
        print("Average Precision:", np.average(precision))
    return accuracy, recall, precision


def plot_confusion(y_true, y_pred, idx=None):
    confusion = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)
    if idx is not None:
        sns.heatmap(confusion[idx[0]:idx[1], idx[0]:idx[1]], cmap='Greens')
    else:
        sns.heatmap(confusion, cmap='Greens')
    plt.tight_layout()
    plt.show()
    plt.close()


def get_class_weight(y):
    sparse = np.argmax(y, axis=1)
    unique, counts = np.unique(sparse, return_counts=True)
    counts = np.column_stack((unique, counts))
    max_freq = np.max(counts[:, 1])
    weights = dict((counts[i][0], max_freq / float(counts[i][1])) for i in range(len(counts)))
    return weights


def get_rare_split(y, threshold=0.5):
    sparse = np.argmax(y, axis=1)
    unique, counts = np.unique(sparse, return_counts=True)
    counts = np.column_stack((unique, counts))
    mean = np.average(counts[:, 1])
    rare_map = dict((counts[i][0], (counts[i][1] < mean*threshold)) for i in range(len(counts)))
    rare = [rare_map[np.argmax(label)] for label in y]
    return rare


def get_datasets(path='./embeddings/half_tenth_train.npz', train_split=0.8):
    x, y, sources = load_train_embed(path)
    x = x.astype('float32')
    y = y.astype('float32')
    p = np.random.permutation(len(x))
    x_shuffle, y_shuffle, = x[p], y[p]
    train_split = int(x.shape[0]*train_split)
    x_train, y_train = x_shuffle[:train_split], y_shuffle[:train_split]
    x_val, y_val = x_shuffle[train_split:], y_shuffle[train_split:]
    train_rares = get_rare_split(y_train, 0.5)
    val_rares = get_rare_split(y_val, 0.5)
    return x_train, y_train, x_val, y_val, train_rares, val_rares


def create_model(verbose=False):
    l1_reg = 0.00
    model = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.5, input_shape=(1280,)),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l1_reg))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.top_k_categorical_accuracy])
    if verbose:
        model.summary()
    return model


def train_model(model, x_train, y_train, x_val, y_val, class_weight, epochs=30, verbose=True):
    BATCH_SIZE = 256
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), class_weight=class_weight, epochs=epochs, batch_size=BATCH_SIZE, verbose=verbose)
    return model


def get_top(y_pred, n=5):
    return np.argsort(y_pred)[:, -n:]


def top_corrected(y_top, y_true):
    scarce_preds = np.zeros((y_true.shape))
    for i, (x, y) in enumerate(zip(y_top, y_true)):
        if y in x:
            scarce_preds[i] = y
        else:
            scarce_preds[i] = x[-1]
    print(scarce_preds)
    return scarce_preds


def plot_recall(recall):
    plt.plot(recall)
    plt.show()
    plt.close()


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, t_rare, v_rare = get_datasets('./embeddings/train_sampled.npz')
    class_weights = get_class_weight(y_train)
    model = create_model()
    model = train_model(model, x_train, y_train, x_val, y_val, class_weight=class_weights)


    y_pred = model.predict(x_val, verbose=False)
    y_pred_sparse = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    print(y_true.shape)
    top5 = get_top(y_pred, 5)
    top5_pred_corrected = top_corrected(top5, y_true)
    print(top5_pred_corrected.shape)

    # x_full, y_full, _, _, _, _ = get_datasets('./embeddings/train_full.npz', train_split=1.0)
    # y_pred = model.predict(x_full, verbose=False)
    # y_pred_sparse = np.argmax(y_pred, axis=1)
    # y_true = np.argmax(y_full, axis=1)
    # print(y_true.shape)
    # top5 = get_top(y_pred, 5)
    # top5_pred_corrected = top_corrected(top5, y_true)
    # print(top5_pred_corrected.shape)

    acc1, recall1, precision1 = get_metrics(y_true=y_true, y_pred=y_pred_sparse, verbose=False)
    acc5, recall5, precision5 = get_metrics(y_true=y_true, y_pred=top5_pred_corrected, verbose=False)
    # plot_recall(recall5)
    # plot_recall(precision5)
    # plot_recall(recall1)
    # plot_recall(precision1)
    # print(recall5)
    print(acc1)
    print(acc5)
    # plot_confusion(y_true=y_true, y_pred=top5_pred_corrected)
    # plot_confusion(y_true=y_true, y_pred=y_pred_sparse)
    # plot_confusion(y_true=y_true[v_rare], y_pred=top5_pred_corrected[v_rare], idx=[100, 200])

    # acc, recall, precision = get_metrics(y_true=y_true[v_rare], y_pred=y_pred[v_rare], idx=[100, 200])
    # plot_confusion(y_true=y_true[v_rare], y_pred=y_pred[v_rare], idx=[100, 200])


    # model = train_model(model, x_train[t_rare], y_train[t_rare], x_val[v_rare], y_val[v_rare], class_weight=class_weights, epochs=100)
