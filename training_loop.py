from data_loader import DataLoader
from evaluation import ModelEvaluation
from model import Model
from active_learner import PoolRanker

if __name__ == "__main__":
    data = DataLoader(pool_path='./embeddings/train_full.npz', train_path="./embeddings/train_sampled.npz",
                      test_path="./embeddings/test.npz")

    model = Model()
    model.set_class_weights(data.y_train)
    model.train(data.x_train, data.y_train, epochs=30)

    pool_learning = PoolRanker(data.x_pool)

    model_evaluator = ModelEvaluation(eval_set_x=data.x_test, eval_set_y=data.y_test)
    model_evaluator.eval_model(model, top_k=5)

    acc, precision, recall = model_evaluator.get_metric_averages(start=0, end=200)
    _, precision_rare, recall_rare = model_evaluator.get_metric_averages(start=100, end=200)
    print(acc)
    print("Total precision: ", precision)
    print("Total recall: ", recall)
    print("Rare precision: ", precision_rare)
    print("Rare recall: ", recall_rare)
    model_evaluator.plot_metrics()
    model_evaluator.plot_confusion()
