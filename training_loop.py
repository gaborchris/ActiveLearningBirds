from data_loader import DataLoader
from evaluation import ModelEvaluation, plot_learning_curves
from model import Model
from active_learner import PoolRanker
import matplotlib.pyplot as plt
import numpy as np


def run_active_learning(data_loader, model, base_path, evaluator, active_method, save_path, curves_path, epochs, output_title, output_path):
    # Start training from base model
    model.load(base_path)
    accuracies1 = []
    precisions1 = []
    recalls1 = []
    accuracies5 = []
    precisions5 = []
    recalls5 = []
    for i in range(epochs):
        # Get active learn label and add to train set
        idx = active_method(model)
        if idx is not None:
            data_loader.append_train_from_pool(idx)

        # Retrain model
        model.set_class_weights(data_loader.y_train)
        model.train(data_loader.x_train, data_loader.y_train, epochs=1, verbose=False)

        # Evaluate model on top 1
        evaluator.eval_model(model, top_k=1, verbose=False)
        a, p, r = evaluator.get_metric_averages(start=0, end=200)
        accuracies1.append(a)
        precisions1.append(p)
        recalls1.append(r)
        print("Epoch", i, "Top 1 -", "Accuracy:", a, "Precision:", p, "Recall:", r)

        # Evaluate model on top 5
        evaluator.eval_model(model, top_k=5, verbose=False)
        a, p, r = evaluator.get_metric_averages(start=0, end=200)
        accuracies5.append(a)
        precisions5.append(p)
        recalls5.append(r)
        print("Epoch", i, "Top 5 -", "Accuracy:", a, "Precision:", p, "Recall:", r)

    np.savez(curves_path, accuracies1, accuracies5, precisions1, precisions5, recalls1, recalls5)
    model.save(save_path)
    create_plots(model, evaluator, save_path=output_path, title=output_title)
    data_loader.reset_train_set()


def run_initial_training(data, model, save_path):
    model.set_class_weights(data.y_train)
    model.train(data.x_train, data.y_train, epochs=30)
    model.save(save_path)


def plot_inital(model, evaluator, base_path, save_path, title):
    model.load(base_path)
    create_plots(model, evaluator, save_path=save_path, title=title)


def create_plots(model, evaluator, save_path, title):
    evaluator.eval_model(model, top_k=5)
    save1 = save_path + "top1"
    save5 = save_path + "top5"
    title1 = title + " Top 1 Predictions"
    title5  = title + " Top 5 Predictions"
    evaluator.plot_confusion(save_path=save5, title=title5)
    evaluator.plot_metrics(save_path=save5, title=title5)

    evaluator.eval_model(model, top_k=1)
    evaluator.plot_confusion(save_path=save1, title=title1)
    evaluator.plot_metrics(save_path=save1, title=title1)


if __name__ == "__main__":
    data = DataLoader(pool_path='./embeddings/train_full.npz', train_path="./embeddings/train_sampled.npz",
                      test_path="./embeddings/test.npz")
    # Paths for each model weights to be saved
    base_path = "./checkpoints/base"
    random_path = "./checkpoints/random"
    least_confidence_path = "./checkpoints/least_conf"
    entropy_path = "./checkpoints/entropy"
    gradient_path = "./checkpoints/gradient"
    full_path = "./checkpoints/full"

    model = Model()
    model_evaluator = ModelEvaluation(eval_set_x=data.x_test, eval_set_y=data.y_test)
    pool_learning = PoolRanker(data.x_pool)

    # run_initial_training(data, model, base_path)
    # plot_inital(model, model_evaluator, base_path, "base", "Base")


    # Active Learn Training for different methods
    active_learn_epochs = 1000
    # run_active_learning(data, model, base_path, model_evaluator, pool_learning.least_condifence,
    #                     save_path=least_confidence_path,
    #                     curves_path="./learning_curves/least_confidence", epochs=active_learn_epochs,
    #                     output_title="Least Confidence", output_path="least_confidence")
    #
    # run_active_learning(data, model, base_path, model_evaluator, pool_learning.max_entropy,
    #                     save_path=entropy_path,
    #                     curves_path="./learning_curves/entropy", epochs=active_learn_epochs,
    #                     output_title="Maximum Entropy", output_path="entropy")

    # run_active_learning(data, model, base_path, model_evaluator, pool_learning.random_select,
    #                     save_path=random_path,
    #                     curves_path="./learning_curves/random", epochs=active_learn_epochs,
    #                     output_title="Random Selection", output_path="random")

    # data.fill_train_set()
    # run_active_learning(data, model, base_path, model_evaluator, pool_learning.none,
    #                     save_path=full_path,
    #                     curves_path="./learning_curves/full", epochs=active_learn_epochs,
    #                     output_title="Full Pool Labeled", output_path="full")
    # rare_start = np.where(np.argmax(data.y_pool, axis=1) == 100)[0][0]
    # print(rare_start)
    # print(np.argmax(data.y_pool[rare_start]))

    plot_learning_curves("./learning_curves")
