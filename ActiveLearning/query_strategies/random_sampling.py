import numpy as np
from .strategy import Strategy, jointStrategy

class RandomSampling(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(RandomSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_drug_idxs==0)[0], n, replace=False)
