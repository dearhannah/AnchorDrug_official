import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class jointStrategy:
    def __init__(self, dataset, net, args_input, args_task):
        self.dataset = dataset
        self.net = net
        self.args_input = args_input
        self.args_task = args_task

    def query(self, n):
        pass
    
    # def get_labeled_count(self):
    #     labeled_idxs, labeled_drugs = self.dataset.get_labeled_drugs()
    #     return len(labeled_idxs)
    
    # def get_model(self):
    #     return self.net.get_model()

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_drug_idxs[pos_idxs] = True
        self.dataset.labeled_data_idxs[self.dataset.drugIdx2dataIdx(pos_idxs)] = True
        if neg_idxs:
            self.dataset.labeled_drug_idxs[neg_idxs] = False
            self.dataset.labeled_data_idxs[self.dataset.drugIdx2dataIdx(pos_idxs)] = False
        print('update drugs')
        print(*pos_idxs, sep = ", ")
        [print(self.dataset.SMILE_train[i]) for i in pos_idxs]

    def train(self, data = None, model_name = None):
        if model_name == None:
            if data == None:
                for i in range(len(self.net)):
                    tmp_net = self.net[i]
                    _, labeled_data = self.dataset.get_labeled_data(dataID=i)
                    tmp_net.train(labeled_data)
            else:
                raise NotImplementedError
                # self.net.train(data)
        else:
            raise NotImplementedError

    def predict(self, id, data):
        preds = self.net[id].predict(data)
        return preds

    def predict_prob(self):
        probs_all = []
        for i in range(len(self.net)):
            tmp_net = self.net[i]
            _, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
            probs = tmp_net.predict_prob(unlabeled_data)
            probs_all.append(probs)
        return probs_all

    # def predict_prob_dropout(self, n_drop=5):
    #     probs_all = []
    #     for i in range(len(self.net)):
    #         tmp_net = self.net[i]
    #         unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
    #         probs = tmp_net.predict_prob_dropout(unlabeled_data, n_drop=n_drop)
    #         probs_all.append(probs)
    #     return probs_all

    def predict_prob_dropout_split(self, n_drop=5):
        probs_all = []
        for i in range(len(self.net)):
            tmp_net = self.net[i]
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
            probs = tmp_net.predict_prob_dropout_split(unlabeled_data, n_drop=n_drop)
            probs_all.append(probs)
        return probs_all
    
    def get_embeddings(self, type='unlabeled'):
        get_embeddings_all = []
        for i in range(len(self.net)):
            tmp_net = self.net[i]
            if type == 'unlabeled':
                _, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
                embeddings = tmp_net.get_embeddings(unlabeled_data)
            elif type == 'all':
                _, data = self.dataset.get_train_data(dataID=i)
                embeddings = tmp_net.get_embeddings(data)
            get_embeddings_all.append(embeddings)
        return get_embeddings_all
    
    def get_grad_embeddings(self, type='unlabeled'):
        get_embeddings_all = []
        for i in range(len(self.net)):
            tmp_net = self.net[i]
            if type == 'unlabeled':
                _, data = self.dataset.get_unlabeled_data(dataID=i)
            elif type == 'all':
                _, data = self.dataset.get_train_data(dataID=i)
            embeddings = tmp_net.get_grad_embeddings(data)
            get_embeddings_all.append(embeddings)
        return get_embeddings_all


