import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init

# pretain_model_pwd = '/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/pretrain_GPS_predictable_307_genes_seed_10_31_final.pth'
# pretain_model_pwd = '/egr/research-aidd/menghan1/AnchorDrug/base_model/hannewnet_1000_256_64/pretrain_GPS_predictable_307_genes_seed_10_39_final.pth'
pretain_model_pwd = '/egr/research-aidd/menghan1/AnchorDrug/base_model/hannewnet_1000_256_64_imbalance/pretrain_GPS_predictable_307_genes_seed_10_36_final.pth'
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.cell = self.params['cell']
        self.device = device
        dim = (2259,)
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
        self.clf.load_state_dict(torch.load(pretain_model_pwd).state_dict())
        
    def train(self, data):
        n_epoch = self.params['n_epoch']
        dim = data.X.shape[1:]
        self.clf = self.net(dim = dim, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
        # cell = self.cell
        self.clf.load_state_dict(torch.load(pretain_model_pwd).state_dict())
        self.clf.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.clf.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, drop_last=False, **self.params['loader_tr_args'])
        loss_record = []
        for epoch in tqdm(range(1, n_epoch+1)):
            loss_epoch = []
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.cpu().item())
            loss_record.append(np.mean(loss_epoch))
        print('finished training')
        print(loss_record)

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, drop_last=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # pred = out.max(1)[1]
                outputs = F.softmax(out, dim=1)
                _, pred = torch.max(outputs.data, 1)
                preds[idxs] = pred.cpu()
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    # def predict_prob_dropout(self, data, n_drop=10):
    #     self.clf.train()
    #     probs = torch.zeros([len(data), len(np.unique(data.Y))])
    #     loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
    #     for i in range(n_drop):
    #         with torch.no_grad():
    #             for x, y, idxs in loader:
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 out, e1 = self.clf(x)
    #                 prob = F.softmax(out, dim=1)
    #                 probs[idxs] += prob.cpu()
    #     probs /= n_drop
    #     return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    # def get_model(self):
    #     return self.clf

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
    
    def get_grad_embeddings(self, data):
        self.clf.eval()
        embDim = self.clf.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = torch.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.clf(x)
                out = out.data.cpu()#.numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu()#.numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
                        else:
                            embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0
        return embeddings

class MLP(nn.Module):
    def __init__(self, dim=(2259,), embSize=64, pretrained=False, num_classes=3, dropout_rate=0.4):
        super(MLP, self).__init__()
        self.dim = embSize
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(dim[0], 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, embSize)
        self.fc4 = nn.Linear(embSize, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        #
    def initial_kaiming_normal(self):
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        #
    def forward(self, x):
        h = x
        h = self.fc1(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.dropout(h)
        h = self.fc3(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.dropout(h)
        logit = self.fc4(h)
        return logit, h
    def get_embedding_dim(self):
        return self.dim