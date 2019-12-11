class GroundTruthOracle:

    def __init__(self, pool_data_x, pool_data_y):
        self.x_data = pool_data_x
        self.y_data = pool_data_y

    def retrieve_label(self, idx):
        return self.y_data[idx]
