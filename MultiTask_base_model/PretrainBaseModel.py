from __future__ import print_function
import random, datetime, wandb
wandb.login()
import os, pickle, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import Normalizer
torch.multiprocessing.set_sharing_strategy('file_system')


def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1') - ord('0')
    return finger_print


class DrugCellline():
    def __init__(self, df, labelname='gene', mode='train'):
        if mode=='train':
            fn = '/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/use_training_cell_line_expression_features_128_encoded_20240111.csv'
        elif mode=='test':
            fn = '/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/test_cell_line_expression_features_128_encoded_20240111.csv'
        cell_map = pd.read_csv(fn, index_col=0)
        self.cell_name = cell_map.index
        transformer = Normalizer().fit(cell_map)
        cell_map = pd.DataFrame(transformer.transform(cell_map), index = cell_map.index, columns = cell_map.columns)
        self.cell_map = cell_map.to_numpy()
        fn = '/egr/research-aidd/menghan1/AnchorDrug/data/drug_fingerprints-1024.csv'
        fp_map = pd.read_csv(fn, header=None, index_col=0)
        self.fp_name = fp_map.index
        self.fp_map = fp_map.to_numpy()

        self.df = df
        print(f"Original DataFrame shape: {df.shape}")
        smiles = df['smiles'].to_list()
        celllines = df['cellline'].to_list()
        labels = torch.from_numpy(np.asarray(df[labelname]))
        # labels = torch.from_numpy(np.asarray(df[labelname]))
        
        # print("get drug features")
        smiles_feature = self.get_drug_fp_batch(smiles).astype(np.float32)
        # print("get cell line features")
        cellline_feature = self.get_cellline_ft_batch(celllines).astype(np.float32)
        data = np.concatenate([smiles_feature, cellline_feature], axis=1)
        
        self.data, self.labels, self.smiles = torch.from_numpy(data), labels.to(torch.long), smiles
    def __getitem__(self, index):
        return self.data[index], self.labels[index], index
    def __len__(self):
        return len(self.data)
    def get_cellline_ft_batch(self, cellline):
        cellline_features = []
        for g in cellline:
            idx = np.where(self.cell_name == g)[0][0]
            cellline_features.append(self.cell_map[idx])
        cellline_features = np.array(cellline_features)
        # print(cellline_features.shape)
        return cellline_features
    def get_drug_fp_batch(self, smile):
        fp_features = []
        for s in smile:
            # print(s)
            try:
                idx = np.where(self.fp_name == s)[0][0]
                fp_features.append(self.fp_map[idx])
            except:
                print(s)
                fp_features.append(get_morgan_fingerprint(s, 3, 1024, FCFP=False))
        fp_features = np.array(fp_features)
        #-------------------------------------------
        #Opt: normalize the ECFP features:
        transformer = Normalizer().fit(fp_features)
        fp_features = transformer.transform(fp_features)
        #-------------------------------------------
        return fp_features


class MLP(nn.Module):
    def __init__(self, input_size=2131, n_outputs=978*3, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_size, 128)
        # self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_outputs)
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
        h = self.fc2(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.fc3(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = self.fc4(h)
        logit = h.view(len(x), 3, 978)
        return logit
    

def eval_metrics(labels_list, preds_list):
    """list of batch labels and batch preds"""
    # labels_flatten = [item for sublist in labels_list for item in sublist]
    labels_flatten = [item for sublist in labels_list for items in sublist for item in items]
    # preds_flatten = [item for sublist in preds_list for item in sublist]
    preds_flatten = [item for sublist in preds_list for items in sublist for item in items]
    # cm = confusion_matrix(labels_flatten, preds_flatten)
    f1 = f1_score(labels_flatten, preds_flatten, average='macro')
    precision = precision_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    recall = recall_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    return f1, precision, recall


def evaluate(model, loader):
    # print('Evaluating')
    model.eval()
    correct = 0
    total = 0
    labels_list = []
    pred_list = []
    for images, labels, indexes in loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        # evaluation
        total += labels.size(0) * labels.size(1)
        correct += (pred == labels).sum()
        pred_list.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    acc = float(correct) / float(total)
    f1, precision, recall = eval_metrics(labels_list, pred_list)
    return acc, f1, precision, recall


def train(model, optimizer, loader):
    # print('Training')
    model.train()
    correct = 0
    total = 0
    labels_list = []
    pred_list = []
    avg_loss = [] #averaged loss across all batches
    for images, labels, indexes in loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # evaluation
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0) * labels.size(1)
        correct += (pred == labels).sum()
        pred_list.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        avg_loss.extend(loss.detach().cpu().numpy().flatten())
    acc = float(correct) / float(total)
    f1, precision, recall = eval_metrics(labels_list, pred_list)
    avg_loss = np.mean(avg_loss)
    return acc, f1, precision, recall, avg_loss


def pretrainEXP(args, df_train, genelist, verbose):
    ## define input data
    train_dataset = DrugCellline(df=df_train, labelname=genelist)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=128, num_workers=0,
        drop_last=False, shuffle=True)
    ## define learner models
    input_size = train_dataset.data.shape[1]
    net = MLP(input_size=input_size)
    net.cuda()
    ## set optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for e in tqdm(range(args.n_epoch)):
        accX, f1X, precisionX, recallX, loss = train(net, optimizer, train_loader)
        if verbose:
            wandb.log({
                "pretrain acc": accX,
                "pretrain f1": f1X,
                "pretrain loss": loss,
                "pretrain precision label 0": precisionX[0],
                "pretrain precision label 1": precisionX[1],
                "pretrain precision label 2": precisionX[2],
                "pretrain recall label 0": recallX[0],
                "pretrain recall label 1": recallX[1],
                "pretrain recall label 2": recallX[2],
                })
        if e%10 == 0:
            torch.save(net.state_dict(), f'/egr/research-aidd/menghan1/AnchorDrug/MultiTask_base_model/pretrain_lr_{args.lr}_epoch_{e}.pt')
    return accX, f1X, precisionX, recallX


def testEXP(args, df_test, genelist, verbose):
    ## define input data
    test_dataset = DrugCellline(df=df_test, labelname=genelist, mode='test')
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=128, num_workers=0,
        drop_last=False, shuffle=False)
    ## define learner models
    input_size = test_dataset.data.shape[1]
    net = MLP(input_size=input_size)
    net.load_state_dict(torch.load(f'/egr/research-aidd/menghan1/AnchorDrug/MultiTask_base_model/pretrain_lr_{args.lr}_epoch_99.pt'))
    net.cuda()
    accY, f1Y, precisionY, recallY = evaluate(net, test_loader)
    return accY, f1Y, precisionY, recallY


def main(args):
    ## Wandb
    wandb.init(
        project='Anchor Drug Project',
        tags = ['PretrainMultiTaskBaseModel'],
        name=f'978gene{args.lr}',
        config=vars(args),
        )
    print(args)
    ## define overall data
    if args.pretrain:
        tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
        cell_map = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/use_training_cell_line_expression_features_128_encoded_20240111.csv', index_col=0)
        cell_name = cell_map.index.to_list()
        tmp = tmp[tmp['cell_iname'].isin(cell_name)]
        tmp = tmp.drop(columns=['sig_id', 'pert_id'])
        tmp.groupby(by=['cell_iname','SMILES']).median().reset_index()
        genelist = tmp.columns.to_list()[2:]
        tmp.loc[:,genelist] = tmp.loc[:,genelist].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
        df_data = tmp.rename(columns={'SMILES': 'smiles', 'cell_iname': 'cellline'})
        verbose = True
        pretrainEXP(args, df_data, genelist, verbose)
    if args.test:
        for cell in ['A549', 'MCF7', 'PC3']:
            print(cell)
            tmp = pd.read_csv(f'/egr/research-aidd/menghan1/AnchorDrug/data/newTestData/{cell}_test.csv', index_col = 'Unnamed: 0')
            tmp.groupby(by=['SMILES']).median().reset_index()
            genelist = tmp.columns.to_list()[1:]
            tmp['cell_iname'] = cell
            tmp.loc[:,genelist] = tmp.loc[:,genelist].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
            df_data = tmp.rename(columns={'SMILES': 'smiles', 'cell_iname': 'cellline'})
            verbose = True
            accY, f1Y, precisionY, recallY = testEXP(args, df_data, genelist, verbose)
            if verbose:
                wandb.log({
                    f"{cell} test acc": accY,
                    f"{cell} test f1": f1Y,
                    f"{cell} test precision label 0": precisionY[0],
                    f"{cell} test precision label 1": precisionY[1],
                    f"{cell} test precision label 2": precisionY[2],
                    f"{cell} test recall label 0": recallY[0],
                    f"{cell} test recall label 1": recallY[1],
                    f"{cell} test recall label 2": recallY[2],
                    })
    wandb.finish()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.005)
    argparser.add_argument('--n_epoch', type=int, help='number of epoch', default=200)
    argparser.add_argument('--pretrain', action='store_true', help='Pretrain or not')
    argparser.add_argument('--test', action='store_true', help='test or not')
    args = argparser.parse_args()
    main(args)