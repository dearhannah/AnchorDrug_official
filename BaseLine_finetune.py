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
    def __init__(self, df, labelname='gene'):
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
        # print(f"Original DataFrame shape: {df.shape}")
        smiles = df['smiles'].to_list()
        celllines = df['cellline'].to_list()
        labels = torch.from_numpy(np.asarray(df[labelname]))
        
        # print("get drug features")
        smiles_feature = self.get_drug_fp_batch(smiles).astype(np.float32)
        # print("get cell line features")
        cellline_feature = self.get_cellline_ft_batch(celllines).astype(np.float32)
        data = np.concatenate([smiles_feature, cellline_feature], axis=1)
        
        self.data, self.labels, self.smiles = torch.from_numpy(data), labels, smiles
        unique, counts = np.unique(self.labels, return_counts=True)
        # print(f"label count: {counts}")
        # print('data shape:')
        # print(self.labels.shape, self.data.shape)
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
    def __init__(self, input_size=2131, n_outputs=3, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
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
        logit = self.fc4(h)
        return logit
    

def eval_metrics(labels_list, preds_list):
    """list of batch labels and batch preds"""
    labels_flatten = [item for sublist in labels_list for item in sublist]
    preds_flatten = [item for sublist in preds_list for item in sublist]
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
        # images = Variable(images).cpu()
        labels = Variable(labels).cuda()
        # labels = Variable(labels).cpu()
        # Forward
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        # evaluation
        total += labels.size(0)
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
    # avg_loss = [] #averaged loss across all batches
    for images, labels, indexes in loader:
        images = Variable(images).cuda()
        # images = Variable(images).cpu()
        labels = Variable(labels).cuda()
        # labels = Variable(labels).cpu()
        # Forward + Backward + Optimize
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # evaluation
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum()
        pred_list.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        # avg_loss.extend(loss.detach().numpy().flatten())
    acc = float(correct) / float(total)
    f1, precision, recall = eval_metrics(labels_list, pred_list)
    # avg_loss = np.mean(avg_loss)
    # print(f"avg_loss:{avg_loss}")
    return acc, f1, precision, recall


def baselineEXP(args, df_train, df_test, verbose):
    genelist = ['HMGCS1', 'TOP2A', 'DNAJB1', 'PCNA', 'HMOX1']
    ResultData = {}
    for g in genelist:
        print(g)
        ## define input data
        labelSETX = list(set(df_train[g].to_list()))
        labelSETY = list(set(df_test[g].to_list()))
        train_dataset = DrugCellline(df=df_train, labelname=g)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=32, num_workers=0,
            drop_last=False, shuffle=True)
        test_dataset = DrugCellline(df=df_test, labelname=g)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=32, num_workers=0,
            drop_last=False, shuffle=False)
        ## define learner models
        input_size = train_dataset.data.shape[1]
        net = MLP(input_size=input_size, n_outputs=3)
        net.cuda()
        if args.pretrain:
            modelPATH = f'/egr/research-aidd/menghan1/AnchorDrug/base_model/pretrain_universal_gene_{g}_seed_10_199_final.pth'
            net.load_state_dict(torch.load(modelPATH).state_dict())
        if args.finetune:
            ## set optimizer
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            for e in tqdm(range(args.n_epoch)):
                accX, f1X, precisionX, recallX = train(net, optimizer, train_loader)
                if len(precisionX)==2:
                    if 0 not in labelSETX:
                        precisionX = np.array([None, precisionX[0], precisionX[1]])
                        recallX = np.array([None, recallX[0], recallX[1]])
                    elif 2 not in labelSETX:
                        precisionX = np.array([precisionX[0], precisionX[1], None])
                        recallX = np.array([recallX[0], recallX[1], None])
                accY, f1Y, precisionY, recallY = evaluate(net, test_loader)
                if len(precisionY)==2:
                    if 0 not in labelSETY:
                        precisionY = np.array([None, precisionY[0], precisionY[1]])
                        recallY = np.array([None, recallY[0], recallY[1]])
                    elif 2 not in labelSETY:
                        precisionY = np.array([precisionY[0], precisionY[1], None])
                        recallY = np.array([recallY[0], recallY[1], None])
                if verbose:
                    wandb.log({
                        # f"{g}: train acc": accX,
                        f"{g}: train f1": f1X,
                        f"{g}: train precision label 0": precisionX[0],
                        f"{g}: train precision label 1": precisionX[1],
                        f"{g}: train precision label 2": precisionX[2],
                        f"{g}: train recall label 0": recallX[0],
                        f"{g}: train recall label 1": recallX[1],
                        f"{g}: train recall label 2": recallX[2],
                        # f"{g}: test acc": accY,
                        f"{g}: test f1": f1Y,
                        f"{g}: test precision label 0": precisionY[0],
                        f"{g}: test precision label 1": precisionY[1],
                        f"{g}: test precision label 2": precisionY[2],
                        f"{g}: test recall label 0": recallY[0],
                        f"{g}: test recall label 1": recallY[1],
                        f"{g}: test recall label 2": recallY[2],
                    })
        acc, f1, precision, recall = evaluate(net, test_loader)
        if len(precision)==2:
            if 0 not in labelSETY:
                precision = np.array([None, precision[0], precision[1]])
                recall = np.array([None, recall[0], recall[1]])
            elif 2 not in labelSETY:
                precision = np.array([precision[0], precision[1], None])
                recall = np.array([recall[0], recall[1], None])
        ResultData[g] = (acc, f1, precision, recall)
        print(acc, f1, precision, recall)
    return ResultData


def main(args):
    ## Wandb
    wandb.init(
        project='Anchor Drug Project',
        tags = ['BaseLine'],
        # tab = ['BaseLine', 'finetune'],
        name=f'{args.cell}',
        config={
            'cellline': args.cell,
            'query':args.querymethod,
            'finetune': args.finetune,
            'pretrain': args.pretrain,
            'epoch': args.n_epoch,
            'lr': args.lr,
        },
        )
    print(args)
    ## define overall data
    genelist = ['HMGCS1', 'TOP2A', 'DNAJB1', 'PCNA', 'HMOX1']
    tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/newTestData/' + args.cell + '_test.csv', index_col = 'Unnamed: 0')
    median = tmp[['SMILES']+genelist].groupby(by='SMILES').median().reset_index()
    median['cellline'] = args.cell
    df_test = median
    df_test = df_test.rename(columns={'SMILES': 'smiles'})
    tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/newTrainData/' + args.cell + '_train.csv', index_col = 'Unnamed: 0')
    median = tmp[['SMILES']+genelist].groupby(by='SMILES').median().reset_index()
    median['cellline'] = args.cell
    df_train = median
    df_train = df_train.rename(columns={'SMILES': 'smiles'})

    for g in genelist:
        df_train[g] = df_train[g].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
        df_test[g] = df_test[g].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)

    ResultData = []
    ResultPath = 'None'
    verbose = True
    if args.querymethod != 'none':
        drugFilePath = f"/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning/druglist/{args.querymethod}/"
        drugFileList = [f for f in os.listdir(drugFilePath) if args.cell in f]
        ##-------------------------------------- this is active learning parameter filter
        drugFileList = [f for f in drugFileList if '_5_0_30' in f]
        for anchor in drugFileList:
            print(anchor)
            with open(drugFilePath+anchor, 'rb') as f:
                druglist = pickle.load(f)
            df_train_anchor = df_train[df_train['smiles'].isin(druglist)]
            ResultData.append(baselineEXP(args, df_train_anchor, df_test, verbose))
            verbose = False
        ResultPath = f"/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/{args.cell}_finetune_{args.querymethod}_5_0_30.pkl"
    else:
        for i in range(3):
            ResultData.append(baselineEXP(args, df_train, df_test, verbose))
            verbose = False
        if args.finetune:
            ResultPath = f"/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/{args.cell}_finetune_all_{args.n_epoch}_{args.lr}.pkl"
        else:
            ResultPath = f"/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/{args.cell}_pretrainOnly.pkl"
    
    wandb.finish()
    with open(ResultPath, 'wb') as f:
	    pickle.dump(ResultData, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()    
    argparser.add_argument('--cell', '-c', type=str, help='cell line', default='MCF7')
    argparser.add_argument('--querymethod', '-q', type=str, help='query method', default='none')
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--n_epoch', type=int, help='number of epoch', default=20)
    argparser.add_argument('--pretrain', action='store_true', help='use pretrained model or not')
    argparser.add_argument('--finetune', action='store_true', help='Finetune or not')
    args = argparser.parse_args()
    main(args)