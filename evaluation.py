import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        print(preds.shape)
        top_preds = get_top(preds, top_k)
        self.preds = top_preds
        self.top_sparse_preds = top_corrected(top_preds, self.eval_y_sparse)
        self.confusion = tf.math.confusion_matrix(labels=self.eval_y_sparse, predictions=self.top_sparse_preds)

    def plot_confusion(self):
        # TODO show x and y label axis and title
        sns.heatmap(self.confusion, cmap='Greens')
        plt.tight_layout()
        plt.show()
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

    def plot_metrics(self):
        # TODO combine graphs and show titles
        self.set_metrics()
        plt.bar(range(len(self.per_class_recall)), self.per_class_recall)
        plt.show()
        plt.close()
        plt.bar(range(len(self.per_class_precision)), self.per_class_precision)
        plt.show()
        plt.close()

    def get_metric_averages(self, start, end):
        self.set_metrics()
        accuracy = self.accuracy
        mean_precision = np.average(self.per_class_precision[start:end])
        mean_recall = np.average(self.per_class_recall[start:end])
        return accuracy, mean_precision, mean_recall

