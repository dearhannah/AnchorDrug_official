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
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

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
    def __init__(self, anchors, model_dict, labels, n):
        fn = '/localscratch2/han/bmdal_reg/data/drug_fingerprints-1024.csv'
        fp_map = pd.read_csv(fn, header=None, index_col=0)
        self.fp_name = fp_map.index
        self.fp_map = fp_map.to_numpy()

        self.anchors = anchors
        self.labels = labels
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
                pred = model(torch.tensor(self.get_drug_fp(a).astype(np.float32)).cuda()).detach()#.numpy()
                # score = np.abs(pred - label[i])[0]
                score = F.mse_loss(pred, label[i])
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
    dir_anchors = args.dir_anchors
    dir_data = args.dir_data
    cell = args.cell_line
    seed = args.seed
    query_method = args.query_method
    initial_size = args.initial_size
    batch_size = args.batch_size
    kernel = args.kernel
    # kernel_transforms = args.kernel_transformation

    wandb.init(project = "trial",
               name = f"{cell}_{str(args.top_n)}_{str(args.subset)}",
               config = args)

    np.random.seed(seed)
    with open(f'{dir_data}{cell}_drug.pkl', 'rb') as f:
        smiles_list = pickle.load(f)
    with open(f'{dir_data}{cell}_x.pkl', 'rb') as f:
        x = pickle.load(f)
    with open(f'{dir_data}{cell}_y.pkl', 'rb') as f:
        y = pickle.load(f)
        
    x = torch.tensor(x).to('cuda')
    y = torch.tensor(y)[:,:].to('cuda')

    idxs_anchors = torch.load(f'{dir_anchors}{cell}_{seed}_{query_method}_{initial_size}_{batch_size}_{kernel}.pt')[:args.subset]
    pool_idxs = torch.tensor([*range(len(smiles_list))], dtype=torch.int64)

    # move new_idxs from the pool set to the training set
    # therefore, we first create a boolean array that is True at the indices in new_idxs and False elsewhere
    logical_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool)
    logical_idxs[idxs_anchors] = True
    # remove them from the pool set
    pool_idxs = pool_idxs[~logical_idxs]

    x_anchor = x[idxs_anchors]
    y_anchor = y[idxs_anchors]
    drug_anchor = [smiles_list[i] for i in idxs_anchors]
    x_pool = x[pool_idxs]
    y_pool = y[pool_idxs]
    drug_pool = [smiles_list[i] for i in pool_idxs]
    
    resouce_cell_list = ['A375', 'HA1E', 'VCAP', 'A549', 'PC3', 'MCF7']
    resouce_cell_list.remove(cell)
    source_models = {}
    for c in resouce_cell_list:
        with open(f'{dir_data}{c}_drug.pkl', 'rb') as f:
            smiles_list_r = pickle.load(f)
        with open(f'{dir_data}{c}_x.pkl', 'rb') as f:
            xi = pickle.load(f)
        with open(f'{dir_data}{c}_y.pkl', 'rb') as f:
            yo = pickle.load(f)
        idxs_train = [i for i in range(len(smiles_list_r)) if smiles_list_r[i] not in smiles_list]
        idxs_train = torch.tensor(idxs_train, dtype=torch.int64)
        xi = xi[idxs_train]
        yo = yo[idxs_train]
        rf_dataset = TensorDataset(torch.tensor(xi), torch.tensor(yo))
        rf_loader = DataLoader(dataset=rf_dataset,
                               batch_size=256, num_workers=0,
                               drop_last=True, shuffle=True)
        model_str = f'/localscratch2/han/bmdal_reg/pretrain_model/pretrain4{cell}_relu_2048_1024.pth'
        model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(),
                              nn.Linear(2048, 1024), nn.ReLU(),
                              nn.Linear(1024, 978)).to('cuda')
        model.load_state_dict(torch.load(model_str))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(20):
            model.train()
            for images, labels in rf_loader:
                images = Variable(images).cuda().to(torch.float32)
                labels = Variable(labels).cuda().to(torch.float32)
                # Forward + Backward + Optimize
                logits = torch.squeeze(model(images))
                loss = F.mse_loss(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
            y_pred = model(x_anchor.cuda())
            loss = ((y_pred - y_anchor.cuda())**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        source_models[c] = copy.deepcopy(model)
        # source_models[c].cpu()


        # test_drugs = list(set(df_finetune['smiles'].to_list())-set(drug_overlap))
        # df_fine_test = df_finetune[df_finetune['smiles'].isin(test_drugs)]
        # df_fine_anchor = df_finetune[df_finetune['smiles'].isin(drug_overlap)]

    anchors_lookup = anchor_finder(drug_pool, drug_anchor)
    model_lookup = model_finder(drug_anchor, source_models, y_anchor, args.top_n)

    anchor_drugs = anchors_lookup.anchors_dict
    predictbles = anchors_lookup.predictbles
    anchor_models = [model_lookup.anchor_models[drug_anchor.index(s)] for s in anchor_drugs]

    pred_list = torch.tensor([]).cuda()
    for i in tqdm(range(len(drug_pool))):
        pred = torch.tensor([]).cuda()
        for n in range(args.top_n):
            model = source_models[anchor_models[i][n]]
            pred = torch.cat((pred, model(x_pool[i])), 0)
            # pred.append(model(x_pool[i]).cpu().detach().numpy()[0])
        pred = torch.mean(pred.reshape(-1, 978), 0)
        pred_list = torch.cat((pred_list, pred), 0)

    rmse = np.sqrt(mean_squared_error(y_pool.flatten().cpu(), pred_list.cpu().detach()))
    corr = pearsonr(y_pool.flatten().cpu(), pred_list.cpu().detach())[0]
    wandb.log({"rmse": rmse, 
               "corr": corr,
               "predictble":len(predictbles)})
    wandb.finish()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dir_data', type=str, help='dir to load data',
                           default='/localscratch2/han/bmdal_reg/data/LINCS_multitask/')
    argparser.add_argument('--out_dir', type=str, help='dir to output',
                           default='/localscratch2/han/bmdal_reg/cell_base_model/')
    argparser.add_argument('--dir_anchors', type=str, help='dir to load anchor drug list',
                           default='/localscratch2/han/bmdal_reg/Druglist_LINCS_AL_with_pretrain_multitask/')    
    argparser.add_argument('--cell_line', type=str, help='cell line', default='MCF7')
    argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.005)
    argparser.add_argument('--top_n', type=int, help='number of cell lines for ensemble', default=4)
    argparser.add_argument('--subset', type=int, help='number of anchor drugs for finetune', default=120)
    argparser.add_argument('--seed', type=int, help='random seed', default=0)
    argparser.add_argument('--query_method', type=str, help='query method', default='random')
    argparser.add_argument('--initial_size', type=int, help='initial_size', default=10)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=10)
    argparser.add_argument('--kernel', type=str, help='kernel', default='grad')
    args = argparser.parse_args()
    main(args)