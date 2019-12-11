class PoolRanker:

    def __init__(self, pool_examples):
        self.pool = pool_examples


    def LeastConfidence(self, model):
        preds = model.predict(self.pool, verbose=False)
        print(preds.shape)