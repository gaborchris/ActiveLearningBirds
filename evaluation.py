import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


def get_top(y_pred, n=5):
    return np.argsort(y_pred)[:, -n:]


def top_corrected(y_top, y_true):
    scarce_preds = np.zeros(y_true.shape)
    for i, (x, y) in enumerate(zip(y_top, y_true)):
        if y in x:
            scarce_preds[i] = y
        else:
            scarce_preds[i] = x[-1]
    return scarce_preds


class ModelEvaluation:

    def __init__(self, eval_set_x, eval_set_y):
        self.eval_set_x = eval_set_x
        self.eval_set_y = eval_set_y
        self.eval_y_sparse = np.argmax(eval_set_y, axis=1)
        self.preds = None
        self.confusion = None
        self.top_sparse_preds = None
        self.per_class_precision = None
        self.per_class_recall = None
        self.accuracy = None

    def eval_model(self, model, verbose=False, top_k=5):
        preds = model.predict(self.eval_set_x, verbose=verbose)
        top_preds = get_top(preds, top_k)
        self.preds = top_preds
        self.top_sparse_preds = top_corrected(top_preds, self.eval_y_sparse)
        self.confusion = tf.math.confusion_matrix(labels=self.eval_y_sparse, predictions=self.top_sparse_preds)

    def plot_confusion(self, save_path, title):
        sns.heatmap(self.confusion, cmap='Greens')
        plt.tight_layout()
        plt.title(title)
        plt.xlabel("predictions")
        plt.ylabel("ground truth labels")
        plt.savefig("./metric_outputs/confusion" + save_path)
        plt.close()

    def set_metrics(self):
        accuracy = np.sum(np.diag(self.confusion)) / np.sum(self.confusion)
        correct = np.diag(self.confusion)
        row_sums = np.sum(self.confusion, axis=1)
        col_sums = np.sum(self.confusion, axis=0)
        recall = np.zeros(shape=correct.shape)
        precision = np.zeros(shape=correct.shape)
        for i, c in enumerate(correct):
            if row_sums[i] != 0:
                recall[i] = c / row_sums[i]
            if col_sums[i] != 0:
                precision[i] = c / col_sums[i]
        self.accuracy = accuracy
        self.per_class_recall = recall
        self.per_class_precision = precision

    def plot_metrics(self, save_path, title):
        self.set_metrics()
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        plt.suptitle('Per Class Precision and Recall for ' + title)

        axs[0].set_title('per class precision')
        axs[0].set_ylabel('precision')
        axs[0].set_xlabel('class label')

        axs[1].set_title('per class recall')
        axs[1].set_ylabel('recall')
        axs[1].set_xlabel('Class Label')

        axs[0].bar(range(len(self.per_class_precision)), self.per_class_precision)
        axs[1].bar(range(len(self.per_class_recall)), self.per_class_recall)

        plt.savefig("./metric_outputs/per_class_" + save_path)
        plt.close()

    def get_metric_averages(self, start, end):
        self.set_metrics()
        accuracy = self.accuracy
        mean_precision = np.average(self.per_class_precision[start:end])
        mean_recall = np.average(self.per_class_recall[start:end])
        return accuracy, mean_precision, mean_recall


def setup_curves_plot():
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].set_title('Mean Class Accuracy')
    axs[1].set_title('Mean Class Precision')
    axs[2].set_title('Mean Class Recall')
    axs[0].set_ylabel('accuracy')
    axs[1].set_ylabel('precision')
    axs[2].set_ylabel('recall')
    axs[0].set_xlabel('active learning step')
    axs[1].set_xlabel('active learning step')
    axs[2].set_xlabel('active learning step')
    return fig, axs


def plot_learning_curves(directory):
    lines = []
    methods = []
    for file in os.listdir(directory):
        filename, _ = os.path.splitext(file)
        methods.append(filename)
        with np.load(os.path.join(directory, file)) as data:
            scores = [x[1] for x in data.items()]
            lines.append(scores)

    fig, axs = setup_curves_plot()
    plt.suptitle("Learning Curves Top 1")
    for method, curves in zip(methods, lines):
        for i in range(3):
            axs[i].plot(curves[2*i], label=method)
            axs[i].legend(prop={'size': 6})
            # axs[i].set_ylim(0, 1)
            # axs[i].set_xlim(0, len(curves[i])-1)
    plt.savefig("./metric_outputs/top1_plots.png")
    plt.close()

    fig, axs = setup_curves_plot()
    plt.suptitle("Learning Curves Top 5")
    for method, curves in zip(methods, lines):
        for i in range(3):
            axs[i].plot(curves[2 * i+1], label=method)
            axs[i].legend(prop={'size': 5})
            # axs[i].set_ylim(0, 1)
            # axs[i].set_xlim(0, len(curves[i])-1)
        print("Before")
        for scoretype in curves:
            print(method, scoretype[0])
        print("After")
        for scoretype in curves:
            print(method, scoretype[-1])
    plt.savefig("./metric_outputs/top5_plots.png")
    plt.close()


