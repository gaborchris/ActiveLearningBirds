import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import load_birds
import buildDataset


IMAGE_SHAPE = (224, 224)


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()


def load_mobilenet():
    classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  # @param {type:"string"}
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
    ])
    return classifier


def imagenet_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels

#
# def test_grace_hopper(classifier, labels):
#     grace_hopper = tf.keras.utils.get_file(
#         'image.jpg',
#         'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
#
#     grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
#     grace_hopper.show()
#     grace_hopper = np.array(grace_hopper) / 255.0
#
#     print(grace_hopper.shape)
#     result = classifier.predict(grace_hopper[np.newaxis, ...])
#     print(result.shape)
#     pred = np.argmax(result[0], axis=-1)
#     print(pred)
#     plt.imshow(grace_hopper)
#     plt.axis('off')
#     pred_name = labels[pred]
#     _ = plt.title("Prediction: "+ pred_name.title())
#     plt.show()
#     plt.close()


def pred_birds(data_root, dataset_name, classifier=None, default_labels=None):
    image_gen = load_birds.get_flow_generator(data_root, dataset_name, batch_size=32, target_size=IMAGE_SHAPE)
    for image_batch, label_batch in image_gen:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break
    result_batch = classifier.predict(image_batch)
    print(result_batch.shape)
    pred_names = default_labels[np.argmax(result_batch, axis=-1)]
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(pred_names[n])
        plt.axis('off')
    plt.show()
    plt.close()


def get_mobilenet_feature_extractor():
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                             input_shape=IMAGE_SHAPE+(3,))
    print('here')
    return feature_extractor_layer


def transer_learn(data_root, dataset_name, base=None, default_labels=None):
    image_gen = load_birds.get_flow_generator(data_root, dataset_name, batch_size=4, target_size=IMAGE_SHAPE, mode='categorical')
    for image_batch, label_batch in image_gen:
        break
    feature_batch = base(image_batch)
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Dense(image_gen.num_classes, activation='softmax')
    ])
    model.summary()
    preds = model(image_batch)
    print(preds.shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss = 'categorical_crossentropy',
        metrics=['acc']
    )
    steps_per_epoch = np.ceil(image_gen.samples/image_gen.batch_size)
    batch_stats_callback = CollectBatchStats()
    history = model.fit_generator(image_gen, epochs=100, steps_per_epoch=steps_per_epoch,
                                  callbacks=[batch_stats_callback])
    plt.figure()
    plt.ylabel("loss")
    plt.xlabel("training steps")
    plt.ylim([0,2])
    plt.plot(batch_stats_callback.batch_losses)
    plt.show()
    plt.close()

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats_callback.batch_acc)
    plt.show()
    plt.close()

    class_names = sorted(image_gen.class_indices.items(), key=lambda pair:pair[1])
    class_names = np.array([key.title() for key, value in class_names])
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]

    for image_batch, label_batch in image_gen:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    label_id = np.argmax(label_batch, axis=-1)
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(4):
        plt.subplot(2, 2, n + 1)
        plt.imshow(image_batch[n])
        color = "green" if predicted_id[n] == label_id[n] else "red"
        plt.title(predicted_label_batch[n].title(), color=color)
        plt.axis('off')
    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()


if __name__ == "__main__":
    # test_grace_hopper(load_mobilenet(), imagenet_labels())
    # test_flowers(load_flowers(), classifier=load_mobilenet(), default_labels=imagenet_labels())
    # pred_birds(buildDataset.DATA_ROOT, 'one_percent', classifier=load_mobilenet(), default_labels=imagenet_labels())
    transer_learn(buildDataset.DATA_ROOT, 'full_data', get_mobilenet_feature_extractor())
