import numpy as np
import torch
import torch.nn.functional as F
from .strategy import jointStrategy
from tqdm import tqdm

class AdversarialBIM(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task, eps=0.0005):
        #eps=0.0001
        super(AdversarialBIM, self).__init__(dataset, net, args_input, args_task)
        self.eps = eps

    def cal_dis(self, tmp_net, x):
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros((1,1024))
        zeros = torch.zeros((1,1235))

        out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1))
        py = out.max(1)[1]
        ny = out.max(1)[1]
        while py.item() == ny.item():
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps * torch.sign(nx.grad.data)[:,:1024]
            nx.grad.data.zero_()

            out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1))
            py = out.max(1)[1]

        return (eta*eta).sum()

    def query(self, n):
        unlabeled_idxs, _ = self.dataset.get_unlabeled_drugs()
        dis_all = []
        for i in range(len(self.net)):
            tmp_net = self.net[i]
            unlabeled_data_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
            tmp_net.clf.cpu()
            tmp_net.clf.eval()
            dis = torch.zeros(unlabeled_data_idxs.shape)
            for j in range(len(unlabeled_data_idxs)):
                x, y, idx = unlabeled_data[j]
                dis[j] = self.cal_dis(tmp_net, x)
            dis = dis.reshape(len(unlabeled_idxs), -1)
            dis_all.append(dis)
            tmp_net.clf.cuda()
        dis_all = torch.concatenate(dis_all, -1)
        dis_all = dis_all.sum(1)

        return unlabeled_idxs[dis_all.argsort()[:n]]
    
    # def cal_dis(self, tmp_net, x):
    #     nx = torch.unsqueeze(x, 0)
    #     nx.requires_grad_()
    #     eta = torch.zeros((1,1024))
    #     zeros = torch.zeros((1,1235))

    #     out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1))
    #     py = out.max(1)[1]
    #     ny = out.max(1)[1]
    #     while py.item() == ny.item():
    #         loss = F.cross_entropy(out, ny)
    #         loss.backward()

    #         eta += self.eps * torch.sign(nx.grad.data)[:,:1024]
    #         nx.grad.data.zero_()

    #         out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1))
    #         py = out.max(1)[1]

    #     return (eta*eta).sum()

    # def query(self, n):
    #     unlabeled_idxs, _ = self.dataset.get_unlabeled_drugs()
    #     dis_all = []
    #     for i in range(len(self.net)):
    #         tmp_net = self.net[i]
    #         unlabeled_data_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
    #         tmp_net.clf.cpu()
    #         tmp_net.clf.eval()
    #         dis = torch.zeros(unlabeled_data_idxs.shape)
    #         for j in range(len(unlabeled_data_idxs)):
    #             x, y, idx = unlabeled_data[j]
    #             dis[j] = self.cal_dis(tmp_net, x)
    #         dis = dis.reshape(len(unlabeled_idxs), -1)
    #         dis_all.append(dis)
    #         tmp_net.clf.cuda()
    #     dis_all = torch.concatenate(dis_all, -1)
    #     dis_all = dis_all.sum(1)

    #     return unlabeled_idxs[dis_all.argsort()[:n]]