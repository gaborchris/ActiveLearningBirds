from model import Model
from evaluation import get_top, top_corrected

import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from buildDataset import get_dir_mapping

def plot_output(image_paths, sparse_labels, top_preds_sparse, top_preds, output_name):
    fig, axs = plt.subplots(5, 2, figsize=(20, 20))

    # plot images
    for i in range(5):
        axs[i][0].imshow(PIL.Image.open(image_paths[i]))
        axs[i][0].set_xticks([])
        axs[i][0].tick_params(labelsize=96)
        axs[i][0].set_yticks([])
        axs[i][0].set_xlabel(label_name_mapping[sparse_labels[i]] + "_" + str(sparse_labels[i]))

        name_pred = [label_name_mapping[x] for x in top_preds_sparse[i]]
        barlist = axs[i][1].bar(name_pred, top_preds[i], color='b')
        if label_name_mapping[sparse_labels[i]] in name_pred:
            loc = np.where(np.array(name_pred) == label_name_mapping[sparse_labels[i]])[0][0]
            barlist[loc].set_color('r')
        if top_preds_sparse[i][-1] == sparse_labels[i]:
            barlist[-1].set_color('g')
        axs[i][1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_name)
    plt.show()
    plt.close()


if __name__ == "__main__":

    model = Model()
    model.load("./checkpoints/base")

    data_path = "./embeddings/test.npz"
    with np.load(data_path) as data:
        x = data['arr_0']
        y = data['arr_1']
        names = data['arr_2']

    # shuffle_data
    # pass

    p = np.random.permutation(len(y))
    names = names[p]
    y = y[p]
    x = x[p]

    images = names[:5]
    y = y[:5]
    x = x[:5]

    image_base_path = "/home/christian/.keras/datasets/bird_test_data/images"
    label_name_mapping = {}
    for i, name in enumerate(sorted(os.listdir(image_base_path))):
        label_name_mapping[i] = name[4:]
    sparse_labels = np.argmax(y, axis=1)
    for i in sparse_labels:
        print(label_name_mapping[i])

    preds = model.predict(x)

    top_preds = np.sort(preds)[:, -5:]
    top_preds_sparse = np.argsort(preds)[:, -5:]
    print(top_preds)
    # top5 = get_top(preds, 5)
    # print(names[:5])
    # print(top5[:5])

    # plot images
    image_paths = []
    for im in images:
        image_paths.append(os.path.join(image_base_path, im))

    plot_output(image_paths, sparse_labels, top_preds_sparse, top_preds, "./bird_output_before.png")

    best_model_path = "./checkpoints/least_conf"
    model.load(best_model_path)
    preds = model.predict(x)
    top_preds = np.sort(preds)[:, -5:]
    top_preds_sparse = np.argsort(preds)[:, -5:]

    plot_output(image_paths, sparse_labels, top_preds_sparse, top_preds, "./bird_output_after.png")



