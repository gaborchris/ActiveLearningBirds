import numpy as np


def load_data(path):
    with np.load(path) as data:
        x = data['arr_0']
        y = data['arr_1']
        n = data['arr_2']
    return x, y


def shuffle_data(data):
    p = np.random.permutation(len(data[1]))
    x_shuffled = data[0][p]
    y_shuffled = data[1][p]
    return x_shuffled, y_shuffled


class DataLoader:

    def __init__(self, train_path, pool_path, test_path):
        self.x_train, self.y_train = shuffle_data(load_data(train_path))
        self.x_pool, self.y_pool = shuffle_data(load_data(pool_path))
        self.x_test, self.y_test = shuffle_data(load_data(test_path))

    def append_train(self, x, y):
        pass


if __name__ == "__main__":
    data_loader = DataLoader(pool_path='./embeddings/train_full.npz', train_path="./embeddings/train_sampled.npz",
               test_path="./embeddings/test.npz")
    print(data_loader.x_train.shape)
    print(data_loader.x_pool.shape)
    print(data_loader.x_test.shape)
