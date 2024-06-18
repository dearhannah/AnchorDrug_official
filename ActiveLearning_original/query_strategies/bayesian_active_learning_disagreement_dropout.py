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
        # a = torch.tensor(([[[j//4+j/10+k/100 for k in range(3)] for j in range(16)] for i in range(5)]))
        
        # probs = [p.reshape(self.n_drop, len(unlabeled_idxs), -1) for p in probs]
        # probs = torch.concatenate(probs,-1)
        # pb = probs.mean(0)
        # entropy1 = (-pb*torch.log(pb)).sum(1)
        # entropy2 = (-probs*torch.log(probs)).sum(2).mean()
        # uncertainties = entropy2 - entropy1
        # return unlabeled_idxs[uncertainties.sort()[1][:n]]
        
        eps = 10e-7
        probs = [p.permute(1,0,2).reshape(-1, self.n_drop, 3) for p in probs]
        probs = torch.concatenate(probs,-1)
        pb = probs.mean(1)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (-probs*torch.log(probs+eps)).sum(2).mean(1)
        uncertainties = entropy2 - entropy1
        uncertainties = uncertainties.reshape(len(unlabeled_idxs), -1).min(1)[0]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
