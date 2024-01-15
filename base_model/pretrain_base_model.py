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

#----------------------------------------------------------------------------------------------------------------
#Changeable parameters:
gene = ['HMGCS1', 'TOP2A', 'DNAJB1', 'PCNA', 'HMOX1', 'MYC'] #The tested gene
c = ['A549', 'MCF7', 'PC3'] #The tested cell line
#----------------------------------------------------------------------------------------------------------------
for g in gene:
    #g = 'HMGCS1' #The tested gene
    #c = ['A549'] #The tested cell line
    type = 'train'
    #type = 'test'
    #Import training & test data:
    x = pd.read_csv('/egr/research-aidd/chenruo4/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
    train_cellline = x['cell_iname'].unique().tolist()
    cellline_map_train = pd.read_csv('/egr/research-aidd/chenruo4/AnchorDrug/cellline_embeddings/training_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
    cellline_map_val = pd.read_csv('/egr/research-aidd/chenruo4/AnchorDrug/cellline_embeddings/val_cell_line_expression_features_128_encoded_20240111.csv', index_col = 'Unnamed: 0')
    cellline_map = pd.concat([cellline_map_train, cellline_map_val])
    use_shared_cellline = list(set(train_cellline) & set(cellline_map.index)) #In the training data, 51 cell lines have CCLE expression features, 63 cell lines do not have them and cannot be trained
    use_cellline_map = cellline_map.loc[use_shared_cellline, :] #51 x 128
    use_cellline_map.to_csv('/egr/research-aidd/chenruo4/AnchorDrug/cellline_embeddings/use_training_cell_line_expression_features_128_encoded_20240111.csv')
#----------------------------------------------------------------------------------------------------------------
    cell_list = use_shared_cellline
    df_data = pd.read_csv('/egr/research-aidd/chenruo4/AnchorDrug/data/pretrainData/resourceCelllines.csv', index_col = 'Unnamed: 0')
    #df_data = df_data.iloc[:, 3:]
    df_data = df_data[df_data['cell_iname'].isin(cell_list)] #17624 rows x 982 columns
    #For replicates, use the median values:
    df_train = None
    for cell in df_data['cell_iname'].unique().tolist():
        tmp = df_data.loc[df_data['cell_iname'] == cell, ['SMILES', g]]
        median = tmp[['SMILES', g,]].groupby(by='SMILES').median().reset_index()
        median['cellline'] = cell
        if df_train is None:
            df_train = median
        else:
            df_train = pd.concat([df_train, median])
    df_train = df_train.rename(columns={g: 'label', 'SMILES': 'smiles'}) #Take the drug-induced gene expression labels of the MYC gene with all drugs
    #
    df_test = None
    for cell in c:
        tmp = pd.read_csv('/egr/research-aidd/chenruo4/AnchorDrug/data/testData/' + cell + '_test.csv', index_col = 'Unnamed: 0')
        median = tmp[['SMILES', g,]].groupby(by='SMILES').median().reset_index()
        median['cellline'] = cell
        if df_test is None:
            df_test = median
        else:
            df_test = pd.concat([df_test, median])
    df_test = df_test.rename(columns={g: 'label', 'SMILES': 'smiles'}) #Take the drug-induced gene expression labels of the MYC gene with all drugs 
#----------------------------------------------------------------------------------------------------------------
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
            fn = '/egr/research-aidd/chenruo4/AnchorDrug/drug_fingerprints-1024.csv'
            fp_map = pd.read_csv(fn, header=None, index_col=0)
            self.fp_name = fp_map.index
            self.fp_map = fp_map.to_numpy()
            self.df = df
            self.random_seed = random_seed
            self.down_sample = down_sample  # training set or test.txt set
            print(df.shape)
            labels = np.asarray(df['label'])
            smiles = df['smiles']
            celllines = df['cellline'] 
            # # be careful, label index need to be reset using np.array
            # quality = np.asarray(df['quality'])
            if self.down_sample:
                idx_in = self.down_sampling(labels)
                smiles = df['smiles'][idx_in]
                celllines = df['cellline'][idx_in]
                labels = np.asarray(df['label'][idx_in])  # be careful, label index need to be reset using np.array
                # quality = np.asarray(df['quality'][idx_in])
            print("get drug features")
            smiles_feature = self.get_drug_fp_batch(smiles).astype(np.float32)
            print("get cell line features")
            cellline_feature = self.get_cellline_ft_batch(celllines).astype(np.float32)
            data = np.concatenate([smiles_feature, cellline_feature], axis=1)
            # self.data, self.labels, self.quality = data, labels, quality
            self.data, self.labels = data, labels
            self.celllines, self.smiles = celllines, smiles
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
            n_select = np.int((np.sum(counts) - max_counts) * 0.5)
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
            self.fc1 = nn.Linear(input_size, 128)
            # self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            # self.bn2 = nn.BatchNorm1d(32)
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
            # h = self.bn1(h)
            h = F.leaky_relu(h, negative_slope=0.01)
            # h = F.dropout(h, p=self.dropout_rate)
            h = self.fc2(h)
            # h = self.bn2(h)
            h = F.leaky_relu(h, negative_slope=0.01)
            # h = F.dropout(h, p=self.dropout_rate)
            h = self.fc3(h)
            h = F.leaky_relu(h, negative_slope=0.01)
            logit = self.fc4(h)
            return logit
#----------------------------------------------------------------------------------------------------------------
    def seed_everything(seed):
        torch.cuda.manual_seed_all(seed)
        #if args.cuda:
        #    torch.cuda.manual_seed_all(seed)
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
    def evaluate(model, loader):
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
            # print("batches")
            #images = Variable(images).cuda()
            images = Variable(images).cpu()
            #labels = Variable(labels).cuda()
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
        for cell in c:
            use_labels_1 = use_labels[count:(count + len(df_test.loc[df_test['cellline'] == cell, :]))]
            use_pred_1 = use_pred[count:(count + len(df_test.loc[df_test['cellline'] == cell, :]))]
            #print(use_labels_1)
            #print('use_label_1 length:')
            #print(len(use_labels_1))
            #print(use_pred_1)
            #print('use_pred_1 length:')
            #print(len(use_pred_1))
            f1_ = f1_score(use_labels_1, use_pred_1, average='macro')
            #cm_, f1_ = eval_metrics(use_labels, use_pred)
            tmp = pd.DataFrame({'pred': use_pred_1, 'truth': use_labels_1})
            acc_ = 100* float(np.sum(tmp['pred'] == tmp['truth'])) / float(len(use_labels_1))
            summary_f1_.extend([f1_*100])
            summary_acc_.extend([acc_])
            #print('my F1 score:')
            #print(100*f1_score(use_labels, use_pred, average='macro'))
            count = count + len(df_test.loc[df_test['cellline'] == cell, :])
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
            #images = Variable(images).cuda()
            images = Variable(images).cpu()
            #labels = Variable(labels).cuda()
            labels = Variable(labels).cpu()
            ## Forward + Backward + Optimize
            logits = model(images)
            #
            #device = torch.device('cpu')
            labels = labels.type(torch.LongTensor)
            #logits, labels = logits.to(device), labels.to(device)
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
        # torch.manual_seed(3407)
        # torch.cuda.manual_seed_all(3407)
        # np.random.seed(3407)
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
            df_train['label'] = df_train['label'].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
            df_test['label'] = df_test['label'].apply(lambda x: (x > 1.5) * 1 + (x >= -1.5) * 1)
            if args.input == 'drugcellline':
                train_dataset = DrugCellline(df=df_train, type = 'train', 
                                        fn_file ='/egr/research-aidd/chenruo4/AnchorDrug/cellline_embeddings/use_training_cell_line_expression_features_128_encoded_20240111.csv', 
                                        down_sample=False)
                test_dataset = DrugCellline(df=df_test, type = 'test', 
                                        fn_file = '/egr/research-aidd/chenruo4/AnchorDrug/cellline_embeddings/test_cell_line_expression_features_128_encoded_20240111.csv', 
                                        down_sample=False)
                #input_size = args.len_fp
                input_size = train_dataset.data.shape[1]
            else:
                print("wrong dataset")
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, num_workers=0,
                                                    drop_last=False, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, num_workers=0,
                                                    drop_last=False, shuffle=False)
            # define learner models
            net = MLP(input_size=input_size, n_outputs=3)
            # if torch.cuda.device_count() > 1:
            #     net = torch.nn.DataParallel(net)
            #net.cuda()
            net.cpu()
            #
            #if opt.cuda:
                #net.cuda()
                #criterion_mse.cuda()
                #x_batch = x_batch.cuda() 
            # check parameters
            print(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            # out setting
            model_str = 'pretrain_universal_'+'gene_' + g + '_seed_' + str(rdseed)
            txtfile = out_dir + model_str + ".txt"
            # rename exsist file for collison
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            if os.path.exists(txtfile):
                os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
            with open(txtfile, "a") as f:
                f.write('epoch,train_acc,train_f1,test_acc,test_f1\n')
            print('epoch,train_acc,train_f1,test_acc,test_f1')
            #
            summary_train_loss = []
            for e in range(args.n_epoch):
                train_acc, train_f1, train_loss = train(net, optimizer, train_loader)
                test_acc, test_f1, test_f1_per_cellline, test_acc_per_cellline = evaluate(net, test_loader)
                print(e, train_acc, train_f1, test_acc, test_f1)
                with open(txtfile, "a") as f:
                    f.write(str(int(e)) + ',' + str(train_acc) + ',' + str(train_f1) + ','
                            + str(test_acc) + ',' + str(test_f1) + '\n')
                if e % 5 == 4:
                    torch.save(net, out_dir + model_str + '_%s.pth' % str(e))
                #print("train_loss:")
                #print(train_loss)
                summary_train_loss.extend([train_loss])
            print("train_loss:")
            print(summary_train_loss)
            pd.DataFrame({'training_loss': summary_train_loss}, index = range(0, e+1)).to_csv(out_dir + g + '_training_loss_per_epoch.csv')
            torch.save(net, out_dir + model_str + '_%s_final.pth' % str(e))
            pd.DataFrame({'test_acc': test_acc_per_cellline, 'test_f1': test_f1_per_cellline, 'cell_line': c}).to_csv(out_dir + g + 'f1_acc_per_test_cellline_final.csv')
#----------------------------------------------------------------------------------------------------------------
    if __name__ == '__main__':
        argparser = argparse.ArgumentParser()
        #argparser.add_argument('--out_dir', type=str, help='dir to output', default='/egr/research-aidd/chenruo4/AnchorDrug/base_model/')
        argparser.add_argument('--out_dir', type=str, help='dir to output', default='/egr/research-aidd/chenruo4/AnchorDrug/base_model/MLP_128_64_32_normalized_cell_line_embeddings_ECFP/')
        argparser.add_argument('--input', type=str, help='input dataset', default='drugcellline')
        argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=0.001)
        argparser.add_argument('--n_epoch', type=int, help='update steps for finetunning', default=200)
        args = argparser.parse_args()
        main(args)

























