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

PretrainModelPath = '/egr/research-aidd/menghan1/AnchorDrug/base_model/hannewnet_1000_128_64_imbalance/pretrain_GPS_predictable_307_genes_seed_10_44_final.pth'

def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1') - ord('0')
    return finger_print


class DrugCelllineGene(data.Dataset):
    def __init__(self, df, cell, balancesample=True, random_seed=random.randint(1,100000)):
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

        fn = '/egr/research-aidd/menghan1/AnchorDrug/data/go_fingerprints_2020.csv'
        gene_map = pd.read_csv(fn)
        self.gene_name = gene_map['gene']
        gene_map = gene_map.drop(columns='gene', axis=1)
        self.gene_map = gene_map.to_numpy()

        self.df = df
        self.random_seed = random_seed
        print(f"Original DataFrame shape: {df.shape}")
        smiles_id = df.index
        genelist = df.columns.to_list()#[:30]
        label_map = df.to_numpy()
        smiles = []
        genes = []
        labels = []
        for s in smiles_id.to_list():
            smiles.extend([s]*len(genelist))
            genes.extend(genelist)
            labels.extend(label_map[np.where(smiles_id == s)[0][0]])

        labels = torch.from_numpy(np.asarray(labels))
        celllines = [cell]*len(smiles)
        
        print("get drug features")
        smiles_feature = self.get_drug_fp_batch(smiles).astype(np.float32)
        print("get cell line features")
        cellline_feature = self.get_cellline_ft_batch(celllines).astype(np.float32)
        print("get gene features")
        gene_feature = self.get_gene_ft_batch(genes).astype(np.float32)
        data = np.concatenate([smiles_feature, cellline_feature, gene_feature], axis=1)
        
        if balancesample:
            balanceIDXs = self.down_sampling(labels)
            data = data[balanceIDXs]
            labels = labels[balanceIDXs]
            
        self.data, self.labels, self.smiles, self.genelist = torch.from_numpy(data), labels, smiles_id, genelist
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"label count: {counts}")
        print('data shape:')
        print(self.labels.shape, self.data.shape)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.labels[index], index
    def __len__(self):
        return len(self.data)
    def down_sampling(self, y):
        unique, counts = np.unique(y, return_counts=True)
        max_idx = np.argmax(counts)
        max_value = unique[max_idx]
        max_counts = counts[max_idx]
        n_select = int((np.sum(counts) - max_counts) * 0.5)
        print('max_value, max_counts, n_select')
        print(max_value, max_counts, n_select)
        random.seed(self.random_seed)
        tmp = list(np.where(y == max_value)[0])
        idx_select = random.sample(tmp, k=n_select)
        idx_select.sort()
        idx_select = np.array(idx_select)
        idx_final = np.concatenate([np.where(y == 0)[0], idx_select, np.where(y == 2)[0]])
        return idx_final
    def get_gene_ft_batch(self, gene):
        gene_features = []
        for g in tqdm(gene):
            idx = np.where(self.gene_name == g)[0][0]
            gene_features.append(self.gene_map[idx])
        gene_features = np.array(gene_features)
        #-------------------------------------------
        #Opt: normalize the GO term features:
        transformer = Normalizer().fit(gene_features)
        gene_features = transformer.transform(gene_features)
        #-------------------------------------------
        # print(gene_features.shape)
        return gene_features
    def get_cellline_ft_batch(self, cellline):
        cellline_features = []
        for g in tqdm(cellline):
            idx = np.where(self.cell_name == g)[0][0]
            cellline_features.append(self.cell_map[idx])
        cellline_features = np.array(cellline_features)
        # print(cellline_features.shape)
        return cellline_features
    def get_drug_fp_batch(self, smile):
        fp_features = []
        for s in tqdm(smile):
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
    def __init__(self, dim=(2259,), embSize=64, num_classes=3, dropout_rate=0.4):
        super(MLP, self).__init__()
        self.dim = embSize
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(dim[0], 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, embSize)
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
        return logit #, h
    def size(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return params.item()
    

def eval_metrics(labels_list, preds_list):
    """list of batch labels and batch preds"""
    labels_flatten = [item for sublist in labels_list for item in sublist]
    preds_flatten = [item for sublist in preds_list for item in sublist]
    cm = confusion_matrix(labels_flatten, preds_flatten)
    f1 = f1_score(labels_flatten, preds_flatten, average='macro')
    precision = precision_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    recall = recall_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    return f1, precision, recall, cm


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
    f1, precision, recall, cm = eval_metrics(labels_list, pred_list)
    return acc, f1, precision, recall, cm, labels_list, pred_list


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
        total += labels.size(0)
        correct += (pred == labels).sum()
        pred_list.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
        avg_loss.extend(loss.detach().cpu().numpy().flatten())
    acc = float(correct) / float(total)
    f1, precision, recall, cm = eval_metrics(labels_list, pred_list)
    avg_loss = np.mean(avg_loss)
    # print(f"avg_loss:{avg_loss}")
    return acc, f1, precision, recall, cm, avg_loss


def baselineEXP(args, df_train, df_test, verbose):
    train_dataset = DrugCelllineGene(df=df_train, cell=args.cell, balancesample=args.balancesample)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batchsize, num_workers=0,
        drop_last=False, shuffle=True)
    test_dataset = DrugCelllineGene(df=df_test, cell=args.cell, balancesample=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batchsize, num_workers=0,
        drop_last=False, shuffle=False)
    net = MLP()
    print("Total number of parameters:", net.size())
    print(net)
    net.cuda()
    metric_history = {}
    
    if args.pretrain:
        net.load_state_dict(torch.load(PretrainModelPath).state_dict())  
    if args.finetune:
        ## set optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        for e in tqdm(range(args.n_epoch)):
            accY, f1Y, precisionY, recallY, cmY, labels_list, pred_list = evaluate(net, test_loader)
            accX, f1X, precisionX, recallX, cmX, avgloss = train(net, optimizer, train_loader)
            if verbose:
                wandb.log({
                    f"train loss": avgloss,
                    f"train acc": accX,
                    f"train f1": f1X,
                    f"train precision label 0": precisionX[0],
                    f"train precision label 1": precisionX[1],
                    f"train precision label 2": precisionX[2],
                    f"train recall label 0": recallX[0],
                    f"train recall label 1": recallX[1],
                    f"train recall label 2": recallX[2],
                    f"test acc": accY,
                    f"test f1": f1Y,
                    f"test precision label 0": precisionY[0],
                    f"test precision label 1": precisionY[1],
                    f"test precision label 2": precisionY[2],
                    f"test recall label 0": recallY[0],
                    f"test recall label 1": recallY[1],
                    f"test recall label 2": recallY[2],
                    })
            metric_history[e] = {
                f"train loss": avgloss,
                f"train acc": accX,
                f"train f1": f1X,
                f"train precision label 0": precisionX[0],
                f"train precision label 1": precisionX[1],
                f"train precision label 2": precisionX[2],
                f"train recall label 0": recallX[0],
                f"train recall label 1": recallX[1],
                f"train recall label 2": recallX[2],
                f"test acc": accY,
                f"test f1": f1Y,
                f"test precision label 0": precisionY[0],
                f"test precision label 1": precisionY[1],
                f"test precision label 2": precisionY[2],
                f"test recall label 0": recallY[0],
                f"test recall label 1": recallY[1],
                f"test recall label 2": recallY[2],
                }
    acc, f1, precision, recall, cm, labels_list, pred_list = evaluate(net, test_loader)
    ResultData = {
        f"acc": acc,
        f"f1": f1,
        f"precision label 0": precision[0],
        f"precision label 1": precision[1],
        f"precision label 2": precision[2],
        f"recall label 0": recall[0],
        f"recall label 1": recall[1],
        f"recall label 2": recall[2],
        }
    return ResultData, cm, metric_history, labels_list, pred_list


def main(args):
    if not args.finetune:
        ResultName = f"{args.cell}_pretrainOnly"
    else:
        ResultName = f"scenario_{str(args.scenario)}_{args.cell}_{args.n_epoch}_{args.lr}_{args.batchsize}"
        if args.querymethod == 'none':
            ResultName = f"{ResultName}_finetune_all"
        else:
            ResultName = f"{ResultName}_finetune_{args.querymethod}_{args.albatch}_0_{args.quota}"
        if args.balancesample:
            ResultName = f"{ResultName}_balancesample"
    print(args)
    
    df_train = pd.read_csv(f'/egr/research-aidd/menghan1/AnchorDrug/data/HQdata/{args.cell}_train.csv', index_col=0)#[1:20]
    df_test = pd.read_csv(f'/egr/research-aidd/menghan1/AnchorDrug/data/HQdata/{args.cell}_test.csv', index_col=0)#[1:20]
    ResultData = {}
    ResultPKG = {}
    verbose = True
    
    if args.querymethod != 'none' and args.anchor:
        if args.scenario == 2:
            drugFileList = [f for f in os.listdir(drugFilePath) if args.cell in f]
        elif args.scenario == 1:
            drugFileList = [f for f in os.listdir(drugFilePath)]
        print('check0')
        print(drugFileList)
        drugFileList = [f for f in drugFileList if args.querymethod in f]
        print('check1')
        print(drugFileList)
        drugFileList = np.sort([f for f in drugFileList if f'_{args.albatch}_0_{args.quota}' in f])
        print('check2')
        print(drugFileList)
        for anchor in drugFileList:
            # Wandb
            wandb.init(
                project='Anchor Drug Project',
                # tags = ['advbim', 'parameter_tuning'],
                tags = ['AL', 'queryscomparison'],
                name=ResultName,
                config=args,
                # {
                #     'cellline': args.cell,
                #     'query':args.querymethod,
                #     'finetune': args.finetune,
                #     'pretrain': args.pretrain,
                #     'balancesample': args.balancesample,
                #     'epoch': args.n_epoch,
                #     'lr': args.lr,
                #     # 'bimratio': args.bimratio,
                #     },
                )
            print(anchor)
            i = int(anchor[-5])
            with open(drugFilePath+anchor, 'rb') as f:
                druglist = pickle.load(f)
            df_train_anchor = df_train.loc[druglist]
            # print('good')
            ResultDatai, cmi, metric_historyi, labels_list, pred_list = baselineEXP(args, df_train_anchor, df_test, verbose)
            ResultData[i] = ResultDatai
            ResultPKG[i] = (ResultDatai, cmi, metric_historyi, labels_list, pred_list)
            # verbose = False
            wandb.finish()
    elif args.querymethod != 'none':
        if args.scenario == 2:
            drugFileList = [f for f in os.listdir(drugFilePath) if f'{args.cell}_{args.quota}' in f]
        elif args.scenario == 1:
            drugFileList = [f for f in os.listdir(drugFilePath) if f'common_{args.quota}' in f]
        for baseline in drugFileList:
            # Wandb
            wandb.init(
                project='Anchor Drug Project',
                # tags = ['advbim', 'parameter_tuning'],
                tags = ['baseline', 'queryscomparison'],
                name=ResultName,
                config=args,
                )
            print(baseline)
            i = int(baseline[-6])
            try:
                druglist = pd.read_csv(drugFilePath+baseline)['drug'].to_list()
            except:
                druglist = pd.read_csv(drugFilePath+baseline)['common_drugs'].to_list()
            df_train_baseline = df_train.loc[druglist]
            # print('good')
            ResultDatai, cmi, metric_historyi, labels_list, pred_list = baselineEXP(args, df_train_baseline, df_test, verbose)
            ResultData[i] = ResultDatai
            ResultPKG[i] = (ResultDatai, cmi, metric_historyi, labels_list, pred_list)
            # verbose = False
            wandb.finish()
    else:
        if args.scenario == 1:
            with open('/egr/research-aidd/menghan1/AnchorDrug/data/HQ_pool_drug.pkl', 'rb') as f:
                trainDrugs = pickle.load(f)
            # print(trainDrugs)
            df_train = df_train.loc[trainDrugs]
        for i in range(3):
            wandb.init(
                project='Anchor Drug Project',
                # tags = ['BaseLine'],
                tags = ['BaseLine', 'finetune', 'all data'],
                name=ResultName,
                config={
                    'cellline': args.cell,
                    'query':args.querymethod,
                    'finetune': args.finetune,
                    'pretrain': args.pretrain,
                    'balancesample': args.balancesample,
                    'epoch': args.n_epoch,
                    'lr': args.lr,
                    },
                )
            ResultDatai, cmi, metric_historyi, labels_list, pred_list = baselineEXP(args, df_train, df_test, verbose)
            ResultData[i] = ResultDatai
            ResultPKG[i] = (ResultDatai, cmi, metric_historyi, labels_list, pred_list)
            # verbose = False
            wandb.finish()

    if not os.path.exists(ResultRoot):
        os.system('mkdir -p %s' % ResultRoot)
        
    with open(f'{ResultRoot}/{ResultName}.pkl', 'wb') as f:
        pickle.dump(ResultPKG, f)
    pd.DataFrame.from_dict(ResultData).to_csv(f'{ResultRoot}/{ResultName}.csv')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()    
    argparser.add_argument('--cell', '-c', type=str, help='cell line', default='MCF7')
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.001)
    argparser.add_argument('--n_epoch', type=int, help='number of epoch', default=50)
    argparser.add_argument('--batchsize', type=int, help='batch size', default=32)
    argparser.add_argument('--pretrain', action='store_true', help='use pretrained model or not')
    argparser.add_argument('--finetune', action='store_true', help='Finetune or not')
    argparser.add_argument('--balancesample', '-bs', action='store_true', help='balance sample or not')
    # Active learning settings
    argparser.add_argument('--scenario', '-s', type=int, default=2, help='scenario 1:common list or 2:cellline specific list')
    argparser.add_argument('--anchor', action='store_true', help='anchor drug method or not')
    argparser.add_argument('--querymethod', '-q', type=str, help='query method', default='none')
    argparser.add_argument('--quota', '-alq', type=int, default=100, help='quota of active learning')
    argparser.add_argument('--albatch', '-alb', type=int, default=10, help='batch size in one active learning iteration')
    # BIM settings
    argparser.add_argument('--bimeps', type=float, default=5e-4, help='learning rate of adv sample')
    argparser.add_argument('--bimdis', type=float, default=0.1, help='distance threshold')
    argparser.add_argument('--bimratio', type=float, default=0.9, help='ratio threshold')
    args = argparser.parse_args()

    # ResultRoot = '/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/new_advBIM_ratio'
    # drugFilePath = '/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning_one_cellline/druglist/new_advbim_ratio/'
    # main(args)
    
    ResultRoot = '/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/all_data'
    main(args)
    
    # ResultRoot = '/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/baselines'
    # for q in ['random', 'clustering']:
    #     args.querymethod = q
    #     drugFilePath = f'/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/{args.querymethod}_selected_drugs/'
    #     args.scenario = 1
    #     for alq in [30, 100]:
    #         args.quota = alq
    #         main(args)
        
    #     args.scenario = 2
    #     drugFilePath = drugFilePath + 'cell_line_specific/'
    #     for alq in [30, 100]:
    #         args.quota = alq
    #         main(args)

    # ResultRoot = '/egr/research-aidd/menghan1/AnchorDrug/resultBaseLine/active_learning'
    # drugFilePath0 = '/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning/druglist/'
    # for c in ['MCF7', 'PC3', 'A549']:
    #     args.cell = c
    #     drugFilePath = drugFilePath0 + 'batch32_epoch20_imbalance/'
    #     args.quota = 100
    #     main(args)
    #     drugFilePath = drugFilePath0 + 'batch32_epoch20_imbalance_30drug/'
    #     args.quota = 30
    #     main(args)
    
    # for query in ['LeastConfidence', 'KCenterGreedy', 'BALDDropout', 'BadgeSampling', 'MarginSampling', 'KMeansSampling', 'RandomSampling']:
    # # for query in ['LeastConfidence', 'BALDDropout', 'MarginSampling']:
    #     args.querymethod =  query
    #     print(args.querymethod)
    #     main(args)
