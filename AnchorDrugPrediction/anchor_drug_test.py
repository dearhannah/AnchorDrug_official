import wandb
wandb.login()

import argparse, os, pickle, yaml, random, copy
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import Normalizer

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    # labels_flatten = [item for sublist in labels_list for item in sublist]
    # preds_flatten = [item for sublist in preds_list for item in sublist]
    labels_flatten = labels_list
    preds_flatten = preds_list
    # cm = confusion_matrix(labels_flatten, preds_flatten)
    f1 = f1_score(labels_flatten, preds_flatten, average='macro', zero_division=0)
    precision = precision_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    recall = recall_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    return f1, precision, recall


def _eval_metrics(labels_list, preds_list):
    """list of batch labels and batch preds"""
    labels_flatten = [item for sublist in labels_list for item in sublist]
    preds_flatten = [item for sublist in preds_list for item in sublist]
    # labels_flatten = labels_list
    # preds_flatten = preds_list
    # cm = confusion_matrix(labels_flatten, preds_flatten)
    f1 = f1_score(labels_flatten, preds_flatten, average='macro', zero_division=0)
    precision = precision_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    recall = recall_score(labels_flatten, preds_flatten, average=None, zero_division=0)
    return f1, precision, recall


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
    f1, precision, recall = _eval_metrics(labels_list, pred_list)
    # avg_loss = np.mean(avg_loss)
    # print(f"avg_loss:{avg_loss}")
    return acc, f1, precision, recall


class anchor_finder():
    def __init__(self, smiles, anchors):
        self.smiles = smiles
        self.anchors = anchors
        self.anchors_dict, self.predictbles = self.get_most_similar_anchor_durg()

    def get_drug_fp_batch(self, smile):
        rdkfp = []
        for s in tqdm(smile):
            ref = Chem.MolFromSmiles(s)
            rdkfp.append(Chem.RDKFingerprint(ref))
        #rdkfp = np.array(rdkfp)
        return rdkfp
    
    def get_most_similar_anchor_durg(self):
        fp = self.get_drug_fp_batch(self.smiles)#.astype(np.float32)
        anchors_fp = self.get_drug_fp_batch(self.anchors)#.astype(np.float32)

        metrics = np.zeros((len(self.smiles), len(self.anchors)))
        indexs = np.zeros(len(self.smiles))

        predictable = []

        for i in range(len(self.smiles)):
            for j in range(len(self.anchors)):
                metrics[i,j] = DataStructs.TanimotoSimilarity(fp[i], anchors_fp[j])
                # metrics[i,j] = pearsonr(fp[i], anchors_fp[j])[0]
            # print(np.argmax(metrics[i]))
            indexs[i] = np.argmax(metrics[i])
            if metrics[i,int(indexs[i])]>=0.85:
                predictable.append(self.smiles[i])
        return [self.anchors[int(i)] for i in indexs], predictable


class model_finder():
    def __init__(self, anchors, model_dict, datas, labels, n):
        fn = '/egr/research-aidd/menghan1/AnchorDrug/data/drug_fingerprints-1024.csv'
        fp_map = pd.read_csv(fn, header=None, index_col=0)
        self.fp_name = fp_map.index
        self.fp_map = fp_map.to_numpy()

        self.anchors = anchors
        self.labels = labels
        self.datas = datas
        self.models = model_dict
        # self.anchor_models = self.get_best_model()
        self.anchor_models, self.models_weight = self.get_top_n_model(n)
    
    def get_drug_fp(self, smile):
        try:
            idx = np.where(self.fp_name == smile)[0][0]
            fp_features = self.fp_map[idx]
        except:
            print(smile)
            fp_features = get_morgan_fingerprint(smile, 3, 1024, FCFP=False)
        fp_features = np.array(fp_features)
        # print(fp_features.shape)
        return fp_features
   
    def get_best_model(self):
        model_dict = self.models
        label = self.labels
        anchors = self.anchors
        models_list = []
        for i, a in enumerate(anchors):
            best_score = 10000
            best = 'notchange'
            for m in model_dict:
                model = model_dict[m].cpu()
                pred = model(torch.tensor(self.get_drug_fp(a).astype(np.float32))).detach().numpy()
                score = np.abs(pred - label[i])
                if score < best_score:
                    best_score = score
                    best = m
            models_list.append(best)
        return models_list
    
    def get_top_n_model(self, n):
        model_dict = self.models
        datas = self.datas
        label = self.labels
        anchors = self.anchors
        models_list = []
        weight_list = []
        for i, a in enumerate(anchors):
            # best_score = 10000
            # best = 'notchange'
            models = []
            scores = []
            for m in model_dict:
                model = model_dict[m]#.cpu()
                # pred = model(torch.tensor(self.get_drug_fp(a).astype(np.float32)).cuda()).detach()#.numpy()
                pred = model(datas[i]).detach()#.numpy()
                # score = np.abs(pred - label[i])[0]
                score = F.cross_entropy(pred, label[i])
                scores.append(score)
                models.append(m)
            sorted_list = sorted(zip(scores,models))[:n]
            tops = [rank[1] for rank in sorted_list]
            weight = [1/rank[0] for rank in sorted_list]
                # if score < best_score:
                #     best_score = score
                #     best = m
            models_list.append(tops)
            weight_list.append(weight)
        return models_list, weight_list


def main(args):
    print(args)
    cell = args.cell_line
    gene = args.gene
    seed = args.seed
    np.random.seed(seed)

    ## Wandb
    wandb.init(
        project='Anchor Drug Project',
        tags = ['AnchorDrugEnsemble'],
        name=f'{cell}_{gene}',
        config={
            'cellline': args.cell_line,
            'gene': args.gene,
            'query':args.querymethod,
            'top_n': args.top_n,
            'seed': args.seed,
            'epoch': args.n_epoch,
            'lr': args.lr,
        },
        )
    ## load test dataset
    tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/newTestData/' + args.cell_line + '_test.csv', index_col = 'Unnamed: 0')
    median = tmp[['SMILES', args.gene]].groupby(by='SMILES').median().reset_index()
    median['cellline'] = args.cell_line
    df_test = median
    df_test[args.gene] = df_test[args.gene].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
    df_test = df_test.rename(columns={'SMILES': 'smiles'})
    test_dataset = DrugCellline(df=df_test, labelname=gene)
    x_pool = test_dataset.data
    y_pool = test_dataset.labels
    drug_pool = test_dataset.smiles
    ## load anchor drug dataset
    tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/newTrainData/' + args.cell_line + '_train.csv', index_col = 'Unnamed: 0')
    median = tmp[['SMILES', args.gene]].groupby(by='SMILES').median().reset_index()
    median['cellline'] = args.cell_line
    df_train = median
    df_train[args.gene] = df_train[args.gene].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
    df_train = df_train.rename(columns={'SMILES': 'smiles'})
    ## anchor drug list loading
    with open(args.path_anchors, 'rb') as f:
        druglist = pickle.load(f)
    ##------------------------------------##
    df_train_anchor = df_train[df_train['smiles'].isin(druglist)]
    train_dataset = DrugCellline(df=df_train_anchor, labelname=gene)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, num_workers=0,
        drop_last=False, shuffle=True)
    x_anchor = train_dataset.data
    y_anchor = train_dataset.labels
    drug_anchor = train_dataset.smiles
    ## prepare ensemble model
    resouce_cell_list = ['OCILY3', 'AN3CA', 'HS578T', 'NOMO1', 'SKBR3', 'HCC515', 'NCIH596', 'U2OS', 'HELA', 'THP1', 'HL60', 'YAPC', 'NCIH1563', 'OC314', '22RV1', 'NCIH1781', 'U937', 'J82', 'MINO', 'NCIH1975', 'K562', 'SNU1040', 'MFE319', 'HCC827', 'MDAMB231', 'BT474', 'VCAP', 'HT29', 'BT20', 'NALM6', 'SUDHL4', 'SKMEL5', 'HCT116', 'JURKAT', 'NCIH2073', 'HA1E', '5637', 'HUH7', 'CW2', 'DV90', 'A375', 'OVCAR8', 'U266B1', 'OCILY19', 'SKNSH', 'NCIH508', 'VMCUB1', 'LN229', 'KMS34', 'HEPG2', 'NCIH2110']
    source_models = {}
    for c in resouce_cell_list:
        model_str = f'/egr/research-aidd/menghan1/AnchorDrug/AnchorDrugPrediction/finetune_step1_models/pretrain_universal_gene_{gene}_cellline_{c}_seed_10_19_final.pth'
        model = MLP(input_size=1152, n_outputs=3).cuda()
        model.load_state_dict(torch.load(model_str).state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for e in tqdm(range(args.n_epoch)):
            accX, f1X, precisionX, recallX = train(model, optimizer, train_loader)
        source_models[c] = copy.deepcopy(model)
        source_models[c].cpu()
    ## ensemble prediction
    anchors_lookup = anchor_finder(drug_pool, drug_anchor)
    model_lookup = model_finder(drug_anchor, source_models, x_anchor, y_anchor, args.top_n)

    anchor_drugs = anchors_lookup.anchors_dict
    predictbles = anchors_lookup.predictbles
    anchor_models = [model_lookup.anchor_models[drug_anchor.index(s)] for s in anchor_drugs]

    # pred_list = torch.tensor([])#.cuda()
    pred_list = []
    for i in tqdm(range(len(drug_pool))):
        # pred = torch.tensor([])#.cuda()
        logits = []
        for n in range(args.top_n):
            model = source_models[anchor_models[i][n]]
            # pred = torch.cat((pred, F.softmax(model(x_pool[i]), dim=0)), 1)
            logits.append(F.softmax(model(x_pool[i]), dim=0))
            # pred.append(model(x_pool[i]).cpu().detach().numpy()[0])
        logits = torch.stack(logits).mean(0)
        _, pred = torch.max(logits.data, 0)
        pred_list.append(pred.item())
    f1, precision, recall = eval_metrics(y_pool.tolist(), pred_list)
    print(f1)
    print(precision)
    print(recall)
    wandb.log({
        f"test f1": f1,
        f"test precision label 0": precision[0],
        f"test precision label 1": precision[1],
        f"test precision label 2": precision[2],
        f"test recall label 0": recall[0],
        f"test recall label 1": recall[1],
        f"test recall label 2": recall[2],
        })
    wandb.finish()
    return f1, precision, recall


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_dir', type=str, help='dir to output',
                           default='/egr/research-aidd/menghan1/AnchorDrug/AnchorDrugPrediction/results/')
    argparser.add_argument('--path_anchors', type=str, help='dir to load anchor drug list',
                           default='/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning/druglist/') 
    argparser.add_argument('--querymethod', '-q', type=str, help='query strategy used, for recording, value not used in code',
                           default='KMeans') 
    argparser.add_argument('--cell_line', '-c', type=str, help='cell line', default='MCF7')
    argparser.add_argument('--gene', type=str, help='gene', default='TOP2A')
    argparser.add_argument('--top_n', type=int, help='number of cell lines for ensemble', default=3)
    argparser.add_argument('--seed', type=int, help='random seed', default=0)
    argparser.add_argument('--n_epoch', type=int, help='epoch', default=10)
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.005)
    args = argparser.parse_args()
    # # # potential needed code: loop for randomnes or anything
    genelist = ['HMGCS1', 'TOP2A', 'DNAJB1', 'PCNA', 'HMOX1']
    default_path = args.path_anchors
    filelist = [f for f in os.listdir(default_path+args.querymethod) if args.cell_line in f]
    filelist = [f for f in filelist if '_5_0_30' in f]
    [print(f) for f in filelist]
    data = []
    for gene in genelist:
        dg = []
        args.gene = gene
        for f in filelist:
            args.path_anchors = f"{default_path}{args.querymethod}/{f}"
            f1, precision, recall = main(args)
            dg.append(np.array([f1] + precision.tolist() + recall.tolist()))
            print(f1, precision, recall)
        data.append(dg)
    with open(f'{args.out_dir}{args.cell_line}_{args.querymethod}.pkl', 'wb') as f:
        pickle.dump(data, f)
