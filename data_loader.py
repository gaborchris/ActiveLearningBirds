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
        self.x_train_default = self.x_train
        self.y_train_default = self.y_train
        self.x_pool, self.y_pool = load_data(pool_path)
        self.x_test, self.y_test = load_data(test_path)

    def append_train_from_pool(self, idx):
        new_x = np.expand_dims(self.x_pool[idx], axis=0)
        new_y = np.expand_dims(self.y_pool[idx], axis=0)
        print("Appending idx", idx, "with label", np.argmax(new_y))
        self.x_train = np.append(self.x_train, new_x, axis=0)
        self.y_train = np.append(self.y_train, new_y, axis=0)
        self.x_train, self.y_train = shuffle_data((self.x_train, self.y_train))

    def reset_train_set(self):
        self.x_train = self.x_train_default
        self.y_train = self.y_train_default

    def fill_train_set(self):
        self.x_train = self.x_pool
        self.y_train = self.y_pool


if __name__ == "__main__":
    data_loader = DataLoader(pool_path='./embeddings/train_full.npz', train_path="./embeddings/train_sampled.npz",
               test_path="./embeddings/test.npz")
    print(data_loader.x_train.shape)
    print(data_loader.y_train.shape)
    # print(data_loader.x_pool.shape)
    # print(data_loader.x_test.shape)
    data_loader.append_train(100)
    print(data_loader.x_train.shape)
    print(data_loader.y_train.shape)

