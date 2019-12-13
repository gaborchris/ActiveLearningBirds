import numpy as np
import random


class PoolRanker:

    def __init__(self, pool_examples):
        self.pool = pool_examples

    def least_condifence(self, model):
        preds = model.predict(self.pool, verbose=False)
        class_confidence = np.max(preds, axis=1)
        lc_idx = np.argmin(class_confidence)
        return lc_idx

    def random_select(self, model):
        # Hardcoded rare start
        rare_start = 4681
        size = self.pool.shape[0]
        if random.random() < 0.1:
            return random.randint(rare_start, size-1)
        else:
            return random.randint(0, rare_start-1)

    def max_entropy(self, model):
        epsilon = 1e-12
        preds = model.predict(self.pool, verbose=False)
        entropy = np.sum(-np.log2(preds+epsilon)*preds, axis=1)
        return np.argmax(entropy)

    def none(self, model):
        return None



