import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import os, pickle
from torchvision import datasets
import torch.utils.data as data
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, confusion_matrix
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

class lincsData:
    def __init__(self, handler, args_task):
        cell_list = args_task['cell']
        self.balancesample = args_task['balancesample']
        self.cell_list = cell_list
        self.handler = handler
        # self.args_task = args_task
        tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/GPS_predictable_genes.csv')
        self.genelist = tmp.x.to_list()
        with open('HQ_pool_drug.pkl', 'rb') as f:
            trainDrugs = pickle.load(f)
        df_data = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/level5_beta_trt_cp_24h_10uM.csv')

        self.SMILE_train, self.X_train, self.Y_train = trainDrugs,[],[]
        self.SMILE_val, self.X_val, self.Y_val = [],[],[]
        for cell in cell_list:
            print(cell)
            use_HQ_sample_id = pd.read_csv(f'/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/revise_use_LINCS_HQ_data_target_cellline_{cell}_sample_id.csv')['x']
            df_target = df_data[df_data['sig_id'].isin(use_HQ_sample_id)]
            median = df_target[['SMILES']+self.genelist].groupby(by='SMILES').median()
            df_target = median

            tmp = pd.read_csv(f'/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/{cell}_internal_val2_data.csv')
            valDrugs = tmp.SMILES.to_list()

            df_train = df_target.loc[trainDrugs]
            df_val = df_target.loc[valDrugs]

            for g in self.genelist:
                df_train[g] = df_train[g].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
                df_val[g] = df_val[g].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)

            raw_train = DrugCelllineGene(df=df_train, cell=cell)
            raw_val = DrugCelllineGene(df=df_val, cell=cell)

            # TODO: add checking the gene_list and the smile_list
            assert self.SMILE_train==raw_train.smiles.to_list(), 'drugs in differernt cell lines not the same'
            self.X_train.append(raw_train.data)
            self.Y_train.append(raw_train.labels)

            self.X_val.append(raw_val.data)
            self.Y_val.append(raw_val.labels)
            self.SMILE_val.append(raw_val.smiles)

        self.n_pool = len(self.SMILE_train)
        self.labeled_data_idxs = np.zeros(self.n_pool*len(self.genelist), dtype=bool)
        self.labeled_drug_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def drugIdx2dataIdx(self, drugidx):
        dataidx = []
        for i in drugidx:
            dataidx.extend([int(i*len(self.genelist)+n) for n in range(len(self.genelist))])
        return np.array(dataidx)
    
    def initialize_labels(self, num):
        # generate initial labeled pool
        if num == 0:
            pass
        else:
            tmp_idxs = np.arange(self.n_pool)
            np.random.shuffle(tmp_idxs)
            self.labeled_drug_idxs[tmp_idxs[:num]] = True
            self.labeled_data_idxs[self.drugIdx2dataIdx(tmp_idxs[:num])] = True

    def get_labeled_data(self, dataID):
        labeled_idxs = np.arange(self.n_pool*len(self.genelist))[self.labeled_data_idxs]
        return labeled_idxs, self.handler(self.X_train[dataID][labeled_idxs], self.Y_train[dataID][labeled_idxs], self.cell_list[dataID], balancesample=self.balancesample)
    
    def get_labeled_drugs(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_drug_idxs]
        return labeled_idxs, [self.SMILE_train[i] for i in labeled_idxs]
    
    def get_unlabeled_data(self, dataID):
        unlabeled_idxs = np.arange(self.n_pool*len(self.genelist))[~self.labeled_data_idxs]
        return unlabeled_idxs, self.handler(self.X_train[dataID][unlabeled_idxs], self.Y_train[dataID][unlabeled_idxs], self.cell_list[dataID])
    
    def get_unlabeled_drugs(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_drug_idxs]
        return unlabeled_idxs, [self.SMILE_train[i] for i in unlabeled_idxs]
    
    def get_train_data(self, dataID):
        return self.labeled_data_idxs.copy(), self.handler(self.X_train[dataID], self.Y_train[dataID], self.cell_list[dataID])
    
    def get_train_drugs(self):
        return self.labeled_drug_idxs.copy(), self.SMILE_train

    def get_test_data(self, dataID):
        return self.handler(self.X_val[dataID], self.Y_val[dataID], self.cell_list[dataID])
    
    def cal_test_acc(self, preds, dataID):
        return 1.0 * (self.Y_val[dataID]==preds).sum().item() / len(preds)
    
    def cal_test_f1(self, preds, dataID):
        return f1_score(self.Y_val[dataID], preds, average='macro')
    
    def cal_test_confusion(self, preds, dataID):
        return confusion_matrix(self.Y_val[dataID], preds)


def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
    m = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
    fp_bits = fp.ToBitString()
    finger_print = np.fromstring(fp_bits, 'u1') - ord('0')
    return finger_print

class DrugCelllineGene():
    def __init__(self, df, cell):
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
        print(f"Original DataFrame shape: {df.shape}")
        smiles_id = df.index
        genelist = df.columns.to_list()
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
        
        self.data, self.labels, self.smiles, self.genelist = torch.from_numpy(data), labels, smiles_id, genelist
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"label count: {counts}")
        print('data shape:')
        print(self.labels.shape, self.data.shape)
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

def get_LINCS(handler, args_task):
    return lincsData(handler, args_task)










