import numpy as np
import torch
from .strategy import Strategy, jointStrategy

class LeastConfidence(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(LeastConfidence, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_drugs = self.dataset.get_unlabeled_drugs()
        probs = self.predict_prob()

        # # 1 max and then mean
        # probs = [p.max(1)[0] for p in probs]
        # uncertainties = torch.mean(torch.stack(probs), dim=0)
        # 2 max from the overall pool
        probs = [p.max(1)[0] for p in probs]
        uncertainties = torch.stack(probs).max(0)[0]
        
        print(uncertainties.sort()[0][:n])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
