import numpy as np
import torch
import torch.nn.functional as F
from .strategy import jointStrategy
from tqdm import tqdm

class AdversarialBIM(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        #eps=0.0001
        super(AdversarialBIM, self).__init__(dataset, net, args_input, args_task)
        print(f'ratio:{args_input.bimratio}')
        self.eps = args_input.bimeps
        self.dis = args_input.bimdis
        self.threshold = args_input.bimdis
        self.ratio = args_input.bimratio
        # self.batchsize = args_input.batch

    def cal_drug_dis(self, tmp_net, nx):
        nx = nx.to(torch.device("cuda"))
        nx.requires_grad_()
        eta = torch.zeros((1,1024)).to(torch.device("cuda"))
        zeros = torch.zeros((1,1235)).to(torch.device("cuda"))

        out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1).repeat(nx.shape[0],1))
        py = out.max(1)[1]
        ny = out.max(1)[1]
        count_iter = 0
        while (sum(py==ny)/len(ny))>=self.ratio and torch.norm(eta)<=self.threshold:
            loss = F.cross_entropy(out, ny)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(nx[:,:1024], 0.001)
            # eta += self.eps * torch.sign(torch.unsqueeze(nx.grad.data.mean(0),0))[:,:1024]
            grad_mean = torch.unsqueeze(nx.grad.data.mean(0),0)
            grad_norm = torch.norm(grad_mean[:,:1024])
            update = grad_mean[:,:1024]/grad_norm
            eta += self.eps * update
            nx.grad.data.zero_()
            out, _ = tmp_net.clf(nx+torch.concatenate((eta,zeros),-1).repeat(nx.shape[0],1))
            py = out.max(1)[1]
            count_iter +=1
        eta = eta.to(torch.device("cpu"))
        if (sum(py==ny)/len(ny))<self.ratio:
            print(torch.norm(eta), count_iter, sum(py==ny)/len(ny), self.threshold)
        return torch.norm(eta), count_iter, sum(py==ny)/len(ny)

    def query(self, n):
        self.threshold = self.dis
        unlabeled_idxs, _ = self.dataset.get_unlabeled_drugs()
        dis_all = []
        iter_number_sum = 0
        tmp_net = self.net[0]
        unlabeled_data_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=0)
        # tmp_net.clf.cuda()
        tmp_net.clf.eval()
        dis = torch.zeros(len(unlabeled_idxs))
        rat = torch.zeros(len(unlabeled_idxs))
        n_gene = int(len(unlabeled_data_idxs)/len(unlabeled_idxs))
        for j in tqdm(range(len(unlabeled_idxs))):
            # print(self.threshold)
            idx_j = np.array([j*n_gene + n for n in range(n_gene)])
            x, y, idx = unlabeled_data[idx_j]
            dis[j], count_iter, rat[j] = self.cal_drug_dis(tmp_net, x)
            iter_number_sum += count_iter
            # dis = dis.reshape(len(unlabeled_idxs), -1)
            if j>=n:
                self.threshold = sorted(dis[:j+1])[n]+0.0001
        dis_all.append(dis)
        dis_all = torch.stack(dis_all, 0).sum(0)
        print(dis_all[dis_all.argsort()[:n]])
        print(rat[dis_all.argsort()[:n]])
        print('sum of update iteration:', iter_number_sum)
        return unlabeled_idxs[dis_all.argsort()[:n]]
        
        # for i in range(len(self.net)):
        #     tmp_net = self.net[i]
        #     unlabeled_data_idxs, unlabeled_data = self.dataset.get_unlabeled_data(dataID=i)
        #     # tmp_net.clf.cuda()
        #     tmp_net.clf.eval()
        #     dis = torch.zeros(len(unlabeled_idxs))
        #     n_gene = int(len(unlabeled_data_idxs)/len(unlabeled_idxs))
        #     for j in tqdm(range(len(unlabeled_idxs))):
        #         idx_j = np.array([j*n_gene + n for n in range(n_gene)])
        #         x, y, idx = unlabeled_data[idx_j]
        #         dis[j], count_iter = self.cal_drug_dis(tmp_net, x)
        #         iter_number_sum += count_iter
        #     # dis = dis.reshape(len(unlabeled_idxs), -1)
        #     dis_all.append(dis)
        #     # tmp_net.clf.cuda()
        # dis_all = torch.stack(dis_all, 0).sum(0)
        # print(dis_all[dis_all.argsort()[:n]])
        # print('sum of update iteration:', iter_number_sum)
        # return unlabeled_idxs[dis_all.argsort()[:n]]
    
                # if j>=self.batchsize:
                #     self.threshold = 