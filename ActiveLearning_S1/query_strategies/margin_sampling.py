import numpy as np
import torch
from .strategy import jointStrategy

class MarginSampling(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(MarginSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_drugs = self.dataset.get_unlabeled_drugs()
        probs = self.predict_prob()

        probs = [p.sort(descending=True)[0] for p in probs]
        probs = [p[:, 0] - p[:,1] for p in probs]
        probs = [p.reshape(len(unlabeled_idxs),-1) for p in probs]

        # # 1 max among all genes, max from all nets
        # probs = [p.max(1)[0] for p in probs]
        # uncertainties = torch.stack(probs).max(0)[0]
        # # 2 max among all genes, mean from all nets
        # probs = [p.max(1)[0] for p in probs]
        # uncertainties = torch.mean(torch.stack(probs), dim=0)
        # # 3 mean among all genes, max from all nets
        # probs = [torch.mean(p, dim=1) for p in probs]
        # uncertainties = torch.stack(probs).max(0)[0]
        # 4 mean among all genes, mean from all nets
        probs = [torch.mean(p, dim=1) for p in probs]
        uncertainties = torch.mean(torch.stack(probs), dim=0)

        print(uncertainties.sort()[0][:n])
        # print(uncertainties.sort()[1][:n])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
