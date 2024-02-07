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
from sklearn.preprocessing import Normalizer
torch.multiprocessing.set_sharing_strategy('file_system')
import copy

#----------------------------------------------------------------------------------------------------------------
#Changeable parameters:
#Import HQ LINCS data:
x = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
use_HQ_sample_id = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/revise_use_LINCS_HQ_data_pretrain_sample_id.csv')['x']            
x.index = x['sig_id']
x = x.loc[use_HQ_sample_id, :] #12300 * 982 
train_cellline = x['cell_iname'].unique().tolist()
cellline_map_train = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/training_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
cellline_map_val = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/val_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
cellline_map = pd.concat([cellline_map_train, cellline_map_val])
use_shared_cellline = list(set(train_cellline) & set(cellline_map.index)) #In the training data, 51 cell lines have CCLE expression features, 63 cell lines do not have them and cannot be trained
use_cellline_map = cellline_map.loc[use_shared_cellline, :] #51 x 128
cell_list = use_shared_cellline #45 cell lines
df_data = x[x['cell_iname'].isin(cell_list)] #7258 * 982

#Import drug pool data:
gene = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/GPS_predictable_genes.csv')['x'].tolist()
c = ['A549', 'MCF7', 'PC3'] #The tested cell line


#----------------------------------------------------------------------------------------------------------------
df_test_GO = None #This is the internal val set 2
df_ext_test_GO = None #This is the test set

for g in gene:
    #Import test data:
#----------------------------------------------------------------------------------------------------------------
    cell_list = use_shared_cellline
    df_data = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
    df_data = df_data[df_data['cell_iname'].isin(cell_list)] #17624 rows x 982 columns
    #For replicates, use the median values:
    df_test = None
    df_ext_test = None
    for cell in c:
        tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/target_cellline_data/' + cell + '_data.csv', index_col = 'Unnamed: 0')
        use_HQ_sample_id = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/revise_use_LINCS_HQ_data_target_cellline_' + cell + '_sample_id.csv')['x']
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
        median2['test_gene'] = g
        median.index = range(0, median.shape[0])
        median['test_gene'] = g
        #
        if df_test is None:
            df_test = median2
        else:
            df_test = pd.concat([df_test, median2])
        #
        if df_ext_test is None:
            df_ext_test = median
        else:
            df_ext_test = pd.concat([df_ext_test, median])
    df_test = df_test.rename(columns={g: 'label', 'SMILES': 'smiles'}) #Take the drug-induced gene expression labels of the MYC gene with all drugs 
    df_ext_test = df_ext_test.rename(columns={g: 'label', 'SMILES': 'smiles'}) #Take the drug-induced gene expression labels of the MYC gene with all drugs 
    #
    if df_test_GO is None:
        df_test_GO = df_test
    else:
        df_test_GO = pd.concat([df_test_GO, df_test])
    #
    if df_ext_test_GO is None:
        df_ext_test_GO = df_ext_test
    else:
        df_ext_test_GO = pd.concat([df_ext_test_GO, df_ext_test])
    print(g)


df_test_GO.index = range(0, df_test_GO.shape[0]) #This is the internal val set 2
df_ext_test_GO.index = range(0, df_ext_test_GO.shape[0]) #This is the test set
#----------------------------------------------------------------------------------------------------------------  
#n_drug_list = [30, 60, 100, 130, 160, 190, 220, 250, 279]
n_drug_list = [100]

anchor_drug_seed_list = [1, 2, 0] 
# anchor_drug_seed_list = [10]


for n_drug in n_drug_list:
    for anchor_drug_seed in anchor_drug_seed_list:
        #Import finetune data (anchor drugs):
        for cell in c: #Each cell line fine-tune one model:
            tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/data/target_cellline_data/' + cell + '_data.csv', index_col = 'Unnamed: 0')
            use_HQ_sample_id = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/revise_use_LINCS_HQ_data_target_cellline_' + cell + '_sample_id.csv')['x']
            tmp.index = tmp['sig_id']
            tmp = tmp.loc[use_HQ_sample_id, :] 
            #tmp = tmp.loc[tmp['cell_iname'] == cell, ['SMILES', g]]
            df_finetune = None
            anchor_drug_file_name_list = [f for f in os.listdir(f'/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning_one_cellline/druglist/drug{n_drug}/') if 'A549' in f]
            anchor_drug_file_name = [f for f in anchor_drug_file_name_list if f'{anchor_drug_seed}.pkl' in f]
            anchor_drug_file_pwd = f'/egr/research-aidd/menghan1/AnchorDrug/ActiveLearning_one_cellline/druglist/drug30/{anchor_drug_file_name[0]}'
            anchor_code = anchor_drug_file_pwd.split('/')[-1].split('.')[0]
            out_dir = f'/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/results/{anchor_code}/'
            print(f'files save to {out_dir}')
            with open(anchor_drug_file_pwd, 'rb') as f:
                anchor_drugs = pickle.load(f)
            for g in gene:
                #median = tmp[['SMILES', g,]].groupby(by='SMILES').median().reset_index()
                median = tmp.loc[tmp['cell_iname'] == cell, ['SMILES', g]][['SMILES', g,]].groupby(by='SMILES').median().reset_index()
                median['cellline'] = cell
                median['test_gene'] = g
                median = median.rename(columns={g: 'label', 'SMILES': 'smiles'})
                #----------------------------------------------------------------------------------------------------------------
                median.index = median['smiles']
        #----------------------------------------------------------------------------------------------------------------
                #Opt: Fine-tune using all drugs as anchor drugs:
                median = median.loc[anchor_drugs, :]
                if df_finetune is None:
                    df_finetune = median
                else:
                    df_finetune = pd.concat([df_finetune, median])
            df_finetune.index = range(0, df_finetune.shape[0])
            #Exclude A from ((all target cell line data) - IV2):
            df_ext_test_GO2 = None
            for d in anchor_drugs:
                if df_ext_test_GO2 is None:
                    df_ext_test_GO2 = df_ext_test_GO.drop(df_ext_test_GO.loc[df_ext_test_GO['smiles'] == d, :].index)
                    df_ext_test_GO2.index = range(0, df_ext_test_GO2.shape[0])
                else:
                    df_ext_test_GO2 = df_ext_test_GO2.drop(df_ext_test_GO2.loc[df_ext_test_GO2['smiles'] == d, :].index)
                    df_ext_test_GO2.index = range(0, df_ext_test_GO2.shape[0])
            #
            #Only use the corresponding cell line as test set:
            df_test_GO_cellline = df_test_GO.loc[df_test_GO['cellline'] == cell, :]
            df_ext_test_GO_cellline = df_ext_test_GO2.loc[df_ext_test_GO2['cellline'] == cell, :]
            #
            def get_morgan_fingerprint(mol, radius, nBits, FCFP=False):
                m = Chem.MolFromSmiles(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=FCFP)
                fp_bits = fp.ToBitString()
                finger_print = np.fromstring(fp_bits, 'u1') - ord('0')
                return finger_print
            #----------------------------------------------------------------------------------------------------------------
            class DrugCellline(data.Dataset):
                def __init__(self, df, type = 'train', fn_file = None, down_sample=True, random_seed=0):
                    fn = fn_file
                    use_cellline_map = pd.read_csv(fn)
                    use_cellline_map = use_cellline_map.rename(columns = {'Unnamed: 0':'cellline'})
                    self.cellline_name = use_cellline_map['cellline']
                    use_cellline_map = use_cellline_map.drop(columns='cellline', axis=1)
                    #-------------------------------------------------
                    #Opt: normalize the cell line embeddings:
                    transformer = Normalizer().fit(use_cellline_map)
                    use_cellline_map = pd.DataFrame(transformer.transform(use_cellline_map), index = use_cellline_map.index, columns = use_cellline_map.columns)
                    #-------------------------------------------------
                    self.use_cellline_map = use_cellline_map.to_numpy()
                    fn = '/egr/research-aidd/menghan1/AnchorDrug/data/drug_fingerprints-1024.csv'
                    fp_map = pd.read_csv(fn, header=None, index_col=0)
                    self.fp_name = fp_map.index
                    self.fp_map = fp_map.to_numpy()
                    GO = '/egr/research-aidd/menghan1/AnchorDrug/data/go_fingerprints_2020.csv'
                    GO_map = pd.read_csv(GO, index_col=0, header = 0)
                    self.GO_name = GO_map.index
                    self.GO_map = GO_map.to_numpy()
                    #
                    self.df = df
                    self.random_seed = random_seed
                    self.down_sample = down_sample  # training set or test.txt set
                    print(df.shape)
                    labels = np.asarray(df['label'])
                    smiles = df['smiles']
                    celllines = df['cellline'] 
                    genes = df['test_gene'] 
                    # # be careful, label index need to be reset using np.array
                    if self.down_sample:
                        idx_in = self.down_sampling(labels)
                        smiles = df['smiles'][idx_in]
                        celllines = df['cellline'][idx_in]
                        genes = df['test_gene'][idx_in]
                        labels = np.asarray(df['label'][idx_in])  # be careful, label index need to be reset using np.array
                    print("get drug features")
                    smiles_feature = self.get_drug_fp_batch(smiles).astype(np.float32)
                    print("get cell line features")
                    cellline_feature = self.get_cellline_ft_batch(celllines).astype(np.float32)
                    print("get gene features") 
                    gene_feature = self.get_gene_ft_batch(genes).astype(np.float32)
                    data = np.concatenate([smiles_feature, cellline_feature, gene_feature], axis=1)
                    # self.data, self.labels, self.quality = data, labels, quality
                    self.data, self.labels = data, labels
                    self.celllines, self.smiles, self.genes = celllines, smiles, genes
                    unique, counts = np.unique(self.labels, return_counts=True)
                    print(counts)
                    print('data shape:')
                    print(self.data.shape)
                    print(self.labels.shape)
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
                def get_cellline_ft_batch(self, cellline):
                    cellline_features = []
                    for g in tqdm(cellline):
                        idx = np.where(self.cellline_name == g)[0][0]
                        cellline_features.append(self.use_cellline_map[idx])
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
                def get_gene_ft_batch(self, test_gene):
                    gene_features = []
                    for g in tqdm(test_gene):
                        idx = np.where(self.GO_name == g)[0][0]
                        gene_features.append(self.GO_map[idx])
                    gene_features = np.array(gene_features)
                    #-------------------------------------------
                    #Opt: normalize the GO term features:
                    transformer = Normalizer().fit(gene_features)
                    gene_features = transformer.transform(gene_features)
                    #-------------------------------------------
                    return gene_features
            #----------------------------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------------------------
            #training model:
            #In shell:
            #export CUDA_VISIBLE_DEVICES=""
            #----------------------------------------------------------------------------------------------------------------
            class MLP(nn.Module):
                def __init__(self, input_size=2131, n_outputs=3, dropout_rate=0.5):
                    super(MLP, self).__init__()
                    self.dropout_rate = dropout_rate
                    self.fc1 = nn.Linear(input_size, 1000)
                    self.fc2 = nn.Linear(1000, 128)
                    self.fc3 = nn.Linear(128, 64)
                    self.fc4 = nn.Linear(64, n_outputs)
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
            #----------------------------------------------------------------------------------------------------------------
            def seed_everything(seed):
                torch.cuda.manual_seed_all(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                torch.backends.cudnn.deterministic = True
            #----------------------------------------------------------------------------------------------------------------
            def eval_metrics(labels_list, preds_list):
                """list of batch labels and batch preds"""
                labels_flatten = [item for sublist in labels_list for item in sublist]
                preds_flatten = [item for sublist in preds_list for item in sublist]
                cm = confusion_matrix(labels_flatten, preds_flatten)
                f1 = f1_score(labels_flatten, preds_flatten, average='macro')
                return cm, f1
            #----------------------------------------------------------------------------------------------------------------
            def evaluate(model, loader, type):
                print('Evaluating')
                model.eval()
                correct = 0
                total = 0
                labels_list = []
                pred_list = []
                use_labels = [] #per cell line
                use_pred = [] #per cell line
                #
                for images, labels, _ in tqdm(loader):
                    images = Variable(images).cpu()
                    labels = Variable(labels).cpu()
                    logits = model(images)
                    outputs = F.softmax(logits, dim=1)
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum()
                    pred_list.append(pred.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    #
                    use_labels.extend(labels.cpu().numpy().flatten())
                    use_pred.extend(pred.cpu().numpy().flatten())
                #Across all cell lines:
                acc = 100 * float(correct) / float(total)
                cm, f1 = eval_metrics(labels_list, pred_list)
                #Per cell line:
                count = 0
                summary_f1_ = []
                summary_acc_ = []
                if type == 'val':
                    for g in gene:
                        use_labels_1 = use_labels[count:(count + len(df_test_GO_cellline.loc[df_test_GO_cellline['test_gene'] == g, :]))]
                        use_pred_1 = use_pred[count:(count + len(df_test_GO_cellline.loc[df_test_GO_cellline['test_gene'] == g, :]))]
                        f1_ = f1_score(use_labels_1, use_pred_1, average='macro')
                        tmp = pd.DataFrame({'pred': use_pred_1, 'truth': use_labels_1})
                        acc_ = 100* float(np.sum(tmp['pred'] == tmp['truth'])) / float(len(use_labels_1))
                        summary_f1_.extend([f1_*100])
                        summary_acc_.extend([acc_])
                        count = count + len(df_test_GO_cellline.loc[df_test_GO_cellline['test_gene'] == g, :])
                #
                if type == 'test':
                    for g in gene:
                        use_labels_1 = use_labels[count:(count + len(df_ext_test_GO_cellline.loc[df_ext_test_GO_cellline['test_gene'] == g, :]))]
                        use_pred_1 = use_pred[count:(count + len(df_ext_test_GO_cellline.loc[df_ext_test_GO_cellline['test_gene'] == g, :]))]
                        f1_ = f1_score(use_labels_1, use_pred_1, average='macro')
                        tmp = pd.DataFrame({'pred': use_pred_1, 'truth': use_labels_1})
                        acc_ = 100* float(np.sum(tmp['pred'] == tmp['truth'])) / float(len(use_labels_1))
                        summary_f1_.extend([f1_*100])
                        summary_acc_.extend([acc_])
                        count = count + len(df_ext_test_GO_cellline.loc[df_ext_test_GO_cellline['test_gene'] == g, :])
                #
                return acc, f1 * 100, summary_f1_, summary_acc_
            #----------------------------------------------------------------------------------------------------------------
            def train(model, optimizer, loader):
                print('Training')
                model.train()
                correct = 0
                total = 0
                labels_list = []
                pred_list = []
                avg_loss = [] #averaged loss across all batches
                for images, labels, indexes in tqdm(loader):
                    images = Variable(images).cpu()
                    labels = Variable(labels).cpu()
                    ## Forward + Backward + Optimize
                    logits = model(images)
                    #
                    labels = labels.type(torch.LongTensor)
                    #
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
                    #
                    avg_loss.extend(loss.detach().numpy().flatten())
                #
                acc = 100 * float(correct) / float(total)
                cm, f1 = eval_metrics(labels_list, pred_list)
                #
                avg_loss = np.mean(avg_loss)
                print("avg_loss:")
                print(avg_loss)
                #
                return acc, f1 * 100, avg_loss
            #----------------------------------------------------------------------------------------------------------------
            def main(args):
                print(args)
                out_dir = args.out_dir
                if not os.path.exists(out_dir):
                    os.system('mkdir -p %s' % out_dir)
                #
                for rdseed in [10]: #set seed = 10
                    print(rdseed)
                    seed_everything(rdseed)
                    #df_train, df_test = split_train_test_random(df_data, rdseed, ratio=100)
                    ## define input data
                    df_finetune['label'] = df_finetune['label'].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
                    df_test_GO_cellline['label'] = df_test_GO_cellline['label'].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
                    df_ext_test_GO_cellline['label'] = df_ext_test_GO_cellline['label'].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
                    #double check:
                    print('df_finetune:')
                    df_finetune
                    print('df_test_GO_cellline:')
                    df_test_GO_cellline
                    print('df_ext_test_GO_cellline:')
                    df_ext_test_GO_cellline
                    #
                    if args.input == 'drugcellline':
                        finetune_dataset = DrugCellline(df=df_finetune, type = 'train', 
                                                    fn_file = '/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/test_cell_line_expression_features_128_encoded_20240111.csv', 
                                                    down_sample=True)
                        test_dataset = DrugCellline(df=df_ext_test_GO_cellline, type = 'test', 
                                                fn_file = '/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/test_cell_line_expression_features_128_encoded_20240111.csv', 
                                                down_sample=False)
                        val_dataset = DrugCellline(df= df_test_GO_cellline, type = 'test', 
                                                fn_file = '/egr/research-aidd/menghan1/AnchorDrug/data/CellLineEncode/test_cell_line_expression_features_128_encoded_20240111.csv', 
                                                down_sample=False)
                        #input_size = args.len_fp
                        input_size = finetune_dataset.data.shape[1]
                    else:
                        print("wrong dataset")
                    train_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=32, num_workers=0,
                                                            drop_last=False, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_dataset.data.shape[0], num_workers=0,
                                                            drop_last=False, shuffle=False)
                    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=val_dataset.data.shape[0], num_workers=0,
                                                            drop_last=False, shuffle=False)
                    # define learner models
                    net = MLP(input_size=input_size, n_outputs=3)
                    net.cpu()
                    #
                    # check parameters
                    print(net)
                    # out setting
                    model_str = 'pretrain_GPS_predictable_307_genes_cellline_' + cell + '_seed_' + str(rdseed)
                    txtfile = out_dir + model_str + ".txt"
                    #txtfile = out_dir + model_str + '_drug_' + str(n_drug) + '_selection_seed_'+ str(anchor_drug_seed) + ".txt"
                    # rename exsist file for collison
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                    if os.path.exists(txtfile):
                        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
                    with open(txtfile, "a") as f:
                        f.write('epoch,train_acc,train_f1,val_acc,val_f1,test_acc,test_f1\n')
                    print('epoch,train_acc, train_f1, val_acc, val_f1, test_acc,test_f1')
                    #
                    #freeze some layers:
                    net.fc1.weight.requires_grad = False
                    net.fc1.bias.requires_grad = False
                    net.fc2.weight.requires_grad = False
                    net.fc2.bias.requires_grad = False
                    #
                    net.load_state_dict(torch.load('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/pretrain_GPS_predictable_307_genes_seed_10_31_final.pth').state_dict())
                    optimizer = torch.optim.Adam(filter(lambda m: m.requires_grad, net.parameters()), lr=args.lr, amsgrad=True, weight_decay=0.001)
                    #
                    summary_train_loss = []
                    for e in range(args.n_epoch):
                        test_acc, test_f1, test_f1_per_gene, test_acc_per_gene = evaluate(net, test_loader, type = 'test')
                        val_acc, val_f1, val_f1_per_gene, val_acc_per_gene = evaluate(net, val_loader, type = 'val')
                        print('test F1:')
                        print(e, test_acc, test_f1)
                        train_acc, train_f1, train_loss = train(net, optimizer, train_loader)
                        print('fine-tuning F1:')
                        print(e, train_acc, train_f1)
                        with open(txtfile, "a") as f:
                            f.write(str(int(e)) + ',' + str(train_acc) + ',' + str(train_f1) + ','
                                    + str(val_acc) + ',' + str(val_f1) + ','
                                   + str(test_acc) + ',' + str(test_f1) + '\n')
                            #f.write(str(int(e)) + ',' 
                             #      + str(val_acc) + ',' + str(val_f1) + ','
                              #     + str(test_acc) + ',' + str(test_f1) + '\n')
                        summary_train_loss.extend([train_loss])
                        with open(txtfile, "a") as f:
                            f.write(str(int(e)) + ',' 
                                    + str(val_acc) + ',' + str(val_f1) + ','
                                    + str(test_acc) + ',' + str(test_f1) + '\n')
                        torch.save(net, out_dir + model_str + '_%s_.pth' % str(e))
                    print("fine-tuning loss:")
                    print(summary_train_loss)
                    pd.DataFrame({'training_loss': summary_train_loss}, index = range(0, e+1)).to_csv(out_dir + cell + '_finetune_loss_per_epoch.csv')
                    torch.save(net, out_dir + model_str + '_%s_final.pth' % str(e))
                    pd.DataFrame({'val_acc': val_acc_per_gene, 'val_f1': val_f1_per_gene, 'gene': gene}).to_csv(out_dir + cell + '_f1_acc_per_val_set_cellline_final.csv')
                    pd.DataFrame({'test_acc': test_acc_per_gene, 'test_f1': test_f1_per_gene, 'gene': gene}).to_csv(out_dir + cell + '_f1_acc_per_test_set_cellline_final.csv')
        #----------------------------------------------------------------------------------------------------------------
            if __name__ == '__main__':
                argparser = argparse.ArgumentParser()  
                argparser.add_argument('--out_dir', type=str, help='dir to output', default=out_dir) 
                argparser.add_argument('--input', type=str, help='input dataset', default='drugcellline')
                argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.001)
                argparser.add_argument('--n_epoch', type=int, help='update steps for finetunning', default=5)
                args = argparser.parse_args()
                main(args)
























