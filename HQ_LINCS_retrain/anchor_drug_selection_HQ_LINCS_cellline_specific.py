from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import argparse, os
import datetime
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error
import torch.multiprocessing
import math
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
torch.multiprocessing.set_sharing_strategy('file_system')

from kmodes.kmodes import KModes
from scipy.stats import mode 
from collections import Counter


#Import HQ LINCS data:
x = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
use_HQ_sample_id = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/revise_use_LINCS_HQ_data_pretrain_sample_id.csv')['x']
x.index = x['sig_id']
x = x.loc[use_HQ_sample_id, :] #12300 * 982 
train_cellline = x['cell_iname'].unique().tolist()
cellline_map_train = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/CellLineEncode/training_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
cellline_map_val = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/CellLineEncode/val_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
cellline_map = pd.concat([cellline_map_train, cellline_map_val])
use_shared_cellline = list(set(train_cellline) & set(cellline_map.index)) #In the training data, 51 cell lines have CCLE expression features, 63 cell lines do not have them and cannot be trained
use_cellline_map = cellline_map.loc[use_shared_cellline, :] #51 x 128
cell_list = use_shared_cellline #45 cell lines
df_data = x[x['cell_iname'].isin(cell_list)] #7258 * 982

#Import drug pool data:
gene = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/GPS_predictable_genes.csv')['x'].tolist()
c = ['A549', 'MCF7', 'PC3'] #The tested cell line

g = gene[0] #This is just for the convenience of code writing.

df_test = None
for cell in c:
    tmp = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/target_cellline_data/' + cell + '_data.csv', index_col = 'Unnamed: 0')
    use_HQ_sample_id = pd.read_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/revise_use_LINCS_HQ_data_target_cellline_' + cell + '_sample_id.csv')['x']
    tmp.index = tmp['sig_id']
    tmp = tmp.loc[use_HQ_sample_id, :] 
    tmp = tmp.loc[tmp['cell_iname'] == cell, ['SMILES', g]]
    median = tmp[['SMILES', g,]].groupby(by='SMILES').median().reset_index()
    median['cellline'] = cell
    #Set aside random 10% data as internal validation set 2:
    median.index = range(0, median.shape[0])
    median = median.sample(frac=1,random_state=4905) #shuffle
    median2 = median.sample(n=round((median.shape[0]/10)),random_state=4905)
    median = median.drop(median2.index)
    #
    median2.index = range(0, median2.shape[0])
    median2.to_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/target_cellline_data/' + cell + '_internal_val2_data.csv')
    #
    if df_test is None:
        df_test = median
    else:
        df_test = pd.concat([df_test, median])

df_test = df_test.rename(columns={g: 'label', 'SMILES': 'smiles'}) #Take the drug-induced gene expression labels of the MYC gene with all drugs 
df_test.index = range(0, df_test.shape[0])
df_test.to_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/data/target_cellline_data/tested_3_celllines_all_data_minus_internal_val2.csv')
#Find all common drugs among the 3 tested cell lines:
common_drugs = None
for cell in c:
    tmp = df_test.loc[df_test['cellline'] == cell, 'smiles'].unique().tolist()
    common_drugs = list(set(df_data['SMILES']) & set(tmp)) 
    pd.DataFrame(common_drugs).to_csv('/home/ubuntu/single_cell/drug induced gene prediction/AnchorDrug/anchor_drug_selection_HQ_LINCS/cellline_specific_drug_pool_' + cell + '.csv')




