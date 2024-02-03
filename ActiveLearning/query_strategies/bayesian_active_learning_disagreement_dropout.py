import numpy as np
import torch
from .strategy import jointStrategy

class BALDDropout(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task, n_drop=5):
        super(BALDDropout, self).__init__(dataset, net, args_input, args_task)
        self.n_drop = n_drop

    def query(self, n):
        unlabeled_idxs, unlabeled_drugs = self.dataset.get_unlabeled_drugs()
        probs = self.predict_prob_dropout_split(n_drop=self.n_drop)
        probs = [p.reshape(self.n_drop, len(unlabeled_idxs), -1) for p in probs]
        probs = torch.concatenate(probs,-1)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
