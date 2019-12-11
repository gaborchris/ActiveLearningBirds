import numpy as np


class PoolRanker:

    def __init__(self, pool_examples):
        self.pool = pool_examples

    def LeastConfidence(self, model):
        preds = model.predict(self.pool, verbose=False)
        class_confidence = np.max(preds, axis=1)
        lc_idx = np.argmin(class_confidence)
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # print(class_confidence)
        # print(class_confidence.shape)
        # print(lc_idx)
        # print(class_confidence[lc_idx])
        # print(preds[lc_idx])
        # print(preds.shape)
        print(self.pool[lc_idx])
        return lc_idx