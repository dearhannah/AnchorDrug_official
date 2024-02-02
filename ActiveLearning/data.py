import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import os, pickle
from torchvision import datasets
import torch.utils.data as data
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, args_task):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        self.args_task = args_task
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_unlabeled_data_by_idx(self, idx):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs][idx]
    
    def get_data_by_idx(self, idx):
        return self.X_train[idx], self.Y_train[idx]

    def get_new_data(self, X, Y):
        return self.handler(X, Y, self.args_task['transform_train'])

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs], self.args_task['transform_train'])
    
    def get_labeled_drugs(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, [self.train_smiles[i] for i in labeled_idxs]
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs], self.args_task['transform_train'])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train, self.args_task['transform_train'])

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test, self.args_task['transform'])
    
    def get_partial_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return self.X_train[labeled_idxs], self.Y_train[labeled_idxs]
    
    def get_partial_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs]

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
    
    def cal_test_f1(self, preds):
        return f1_score(self.Y_test, preds, average='weighted')


class lincsData:
    def __init__(self, handler, args_task):
        cell_list = args_task['cell']
        self.cell_list = cell_list
        self.handler = handler
        # self.args_task = args_task
        tmp = pd.read_csv('/egr/research-aidd/menghan1/AnchorDrug/HQ_LINCS_retrain/GPS_predictable_genes.csv')
        self.genelist = tmp.x.to_list()#[:3]
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
    
    # def get_unlabeled_data_by_idx(self, dataID, idx):
    #     unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
    #     return self.X_train[unlabeled_idxs][idx]
    
    # def get_data_by_idx(self, dataID, idx):
    #     return self.X_train[idx], self.Y_train[dataID][idx]

    # def get_new_data(self, X, Y):
    #     return self.handler(X, Y)

    def get_labeled_data(self, dataID):
        labeled_idxs = np.arange(self.n_pool*len(self.genelist))[self.labeled_data_idxs]
        return labeled_idxs, self.handler(self.X_train[dataID][labeled_idxs], self.Y_train[dataID][labeled_idxs], self.cell_list[dataID])
    
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
    
    # def get_partial_labeled_data(self, dataID):
    #     labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
    #     return self.X_train[labeled_idxs], self.Y_train[dataID][labeled_idxs]
    
    # def get_partial_unlabeled_data(self, dataID):
    #     unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
    #     return self.X_train[unlabeled_idxs], self.Y_train[dataID][unlabeled_idxs]

    def cal_test_acc(self, preds, dataID):
        return 1.0 * (self.Y_val[dataID]==preds).sum().item() / len(preds)
    
    def cal_test_f1(self, preds, dataID):
        return f1_score(self.Y_val[dataID], preds, average='macro')


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
    
def get_MNIST(handler, args_task):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_FashionMNIST(handler, args_task):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_EMNIST(handler, args_task):
    raw_train = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=True, download=True)
    raw_test = datasets.EMNIST('./data/EMNIST', split = 'byclass', train=False, download=True)
    return Data(raw_train.data, raw_train.targets, raw_test.data, raw_test.targets, handler, args_task)

def get_SVHN(handler, args_task):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data, torch.from_numpy(data_train.labels), data_test.data, torch.from_numpy(data_test.labels), handler, args_task)

def get_CIFAR10(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_CIFAR10_imb(handler, args_task):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    X_tr = data_train.data
    Y_tr = torch.from_numpy(np.array(data_train.targets)).long()
    X_te = data_test.data
    Y_te = torch.from_numpy(np.array(data_test.targets)).long()
    ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    X_tr_imb = []
    Y_tr_imb = []
    random.seed(4666)
    for i in range(Y_tr.shape[0]):
        tmp = random.random()
        if tmp < ratio[Y_tr[i]]:
            X_tr_imb.append(X_tr[i])
            Y_tr_imb.append(Y_tr[i])
    X_tr_imb = np.array(X_tr_imb).astype(X_tr.dtype)
    Y_tr_imb = torch.LongTensor(np.array(Y_tr_imb)).type_as(Y_tr)
    return Data(X_tr_imb, Y_tr_imb, X_te, Y_te, handler, args_task)

def get_CIFAR100(handler, args_task):
    data_train = datasets.CIFAR100('./data/CIFAR100', train=True, download=True)
    data_test = datasets.CIFAR100('./data/CIFAR100', train=False, download=True)
    return Data(data_train.data, torch.LongTensor(data_train.targets), data_test.data, torch.LongTensor(data_test.targets), handler, args_task)

def get_TinyImageNet(handler, args_task):
    import cv2
    #download data from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it into ./data/TinyImageNet
    # deal with training set
    Y_train_t = []
    train_img_names = []
    train_imgs = []
    
    with open('./data/TinyImageNet/tiny-imagenet-200/wnids.txt') as wnid:
        for line in wnid:
            Y_train_t.append(line.strip('\n'))
    for Y in Y_train_t:
        Y_path = './data/TinyImageNet/tiny-imagenet-200/train/' + Y + '/' + Y + '_boxes.txt'
        train_img_name = []
        with open(Y_path) as Y_p:
            for line in Y_p:
                train_img_name.append(line.strip('\n').split('\t')[0])
        train_img_names.append(train_img_name)
    train_labels = np.arange(200)
    idx = 0
    for Y in Y_train_t:
        train_img = []
        for img_name in train_img_names[idx]:
            img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/train/', Y, 'images', img_name)
            train_img.append(cv2.imread(img_path))
        train_imgs.append(train_img)
        idx = idx + 1
    train_imgs = np.array(train_imgs)
    train_imgs = train_imgs.reshape(-1, 64, 64, 3)
    X_tr = []
    Y_tr = []
    for i in range(train_imgs.shape[0]):
        Y_tr.append(i//500)
        X_tr.append(train_imgs[i])
    #X_tr = torch.from_numpy(np.array(X_tr))
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #deal with testing (val) set
    Y_test_t = []
    Y_test = []
    test_img_names = []
    test_imgs = []
    with open('./data/TinyImageNet/tiny-imagenet-200/val/val_annotations.txt') as val:
        for line in val:
            test_img_names.append(line.strip('\n').split('\t')[0])
            Y_test_t.append(line.strip('\n').split('\t')[1])
    for i in range(len(Y_test_t)):
        for i_t in range(len(Y_train_t)):
            if Y_test_t[i] == Y_train_t[i_t]:
                Y_test.append(i_t)
    test_labels = np.array(Y_test)
    test_imgs = []
    for img_name in test_img_names:
        img_path = os.path.join('./data/TinyImageNet/tiny-imagenet-200/val/images', img_name)
        test_imgs.append(cv2.imread(img_path))
    test_imgs = np.array(test_imgs)
    X_te = []
    Y_te = []

    for i in range(test_imgs.shape[0]):
        X_te.append(test_imgs[i])
        Y_te.append(Y_test[i])
    #X_te = torch.from_numpy(np.array(X_te))
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_openml(handler, args_task, selection = 6):
    import openml
    from sklearn.preprocessing import LabelEncoder
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./data/openml/')
    ds = openml.datasets.get_dataset(selection)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    num_classes = int(max(y) + 1)
    nSamps, _ = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split =int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == num_classes: break
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_BreakHis(handler, args_task):
    # download data from https://www.kaggle.com/datasets/ambarish/breakhis and unzip it in data/BreakHis/
    data_dir = './data/BreakHis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast'
    data = datasets.ImageFolder(root = data_dir, transform = None).imgs
    train_ratio = 0.7
    test_ratio = 0.3
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    train_idx = data_idx[:int(len(data)*train_ratio)]
    test_idx = data_idx[int(len(data)*train_ratio):]
    X_tr = [np.array(Image.open(data[i][0])) for i in train_idx]
    Y_tr = [data[i][1] for i in train_idx]
    X_te = [np.array(Image.open(data[i][0])) for i in test_idx]
    Y_te = [data[i][1] for i in test_idx]
    X_tr = np.array(X_tr, dtype=object)
    X_te = np.array(X_te, dtype=object)
    Y_tr = torch.from_numpy(np.array(Y_tr))
    Y_te = torch.from_numpy(np.array(Y_te))
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_PneumoniaMNIST(handler, args_task):
    # download data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and unzip it in data/PhwumniaMNIST/
    import cv2

    data_train_dir = './data/PneumoniaMNIST/chest_xray/train/'
    data_test_dir = './data/PneumoniaMNIST/chest_xray/test/'
    assert os.path.exists(data_train_dir)
    assert os.path.exists(data_test_dir)

    #train data
    train_imgs_path_0 = [data_train_dir+'NORMAL/'+f for f in os.listdir(data_train_dir+'/NORMAL/')]
    train_imgs_path_1 = [data_train_dir+'PNEUMONIA/'+f for f in os.listdir(data_train_dir+'/PNEUMONIA/')]
    train_imgs_0 = []
    train_imgs_1 = []
    for p in train_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_0.append(im)
    for p in train_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        train_imgs_1.append(im)
    train_labels_0 = np.zeros(len(train_imgs_0))
    train_labels_1 = np.ones(len(train_imgs_1))
    X_tr = []
    Y_tr = []
    train_imgs = train_imgs_0 + train_imgs_1
    train_labels = np.concatenate((train_labels_0, train_labels_1))
    idx_train = list(range(len(train_imgs)))
    random.seed(4666)
    random.shuffle(idx_train)
    X_tr = [train_imgs[i] for i in idx_train]
    Y_tr = [train_labels[i] for i in idx_train]
    X_tr = np.array(X_tr)
    Y_tr = torch.from_numpy(np.array(Y_tr)).long()

    #test data
    test_imgs_path_0 = [data_test_dir+'NORMAL/'+f for f in os.listdir(data_test_dir+'/NORMAL/')]
    test_imgs_path_1 = [data_test_dir+'PNEUMONIA/'+f for f in os.listdir(data_test_dir+'/PNEUMONIA/')]
    test_imgs_0 = []
    test_imgs_1 = []
    for p in test_imgs_path_0:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_0.append(im)
    for p in test_imgs_path_1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (224, 224))
        test_imgs_1.append(im)
    test_labels_0 = np.zeros(len(test_imgs_0))
    test_labels_1 = np.ones(len(test_imgs_1))
    X_te = []
    Y_te = []
    test_imgs = test_imgs_0 + test_imgs_1
    test_labels = np.concatenate((test_labels_0, test_labels_1))
    idx_test = list(range(len(test_imgs)))
    random.seed(4666)
    random.shuffle(idx_test)
    X_te = [test_imgs[i] for i in idx_test]
    Y_te = [test_labels[i] for i in idx_test]
    X_te = np.array(X_te)
    Y_te = torch.from_numpy(np.array(Y_te)).long()

    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)

def get_waterbirds(handler, args_task):
    import wilds
    from torchvision import transforms
    dataset = wilds.get_dataset(dataset='waterbirds', root_dir='./data/waterbirds', download='True')
    trans = transforms.Compose([transforms.Resize([255,255])])
    train = dataset.get_subset(split = 'train',transform = trans)
    test = dataset.get_subset(split = 'test', transform = trans)

    len_train = train.metadata_array.shape[0]
    len_test = test.metadata_array.shape[0]
    X_tr = []
    Y_tr = []
    X_te = []
    Y_te = []

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    f = open('waterbirds.txt', 'w')

    for i in range(len_train):
        x,y,meta = train.__getitem__(i)
        img = np.array(x)
        X_tr.append(img)
        Y_tr.append(y)

    for i in range(len_test):
        x,y, meta = test.__getitem__(i)
        img = np.array(x)

        X_te.append(img)
        Y_te.append(y)
        if meta[0] == 0 and meta[1] == 0:
            f.writelines('1') #landbird_background:land
            f.writelines('\n')
            count1 = count1 + 1
        elif meta[0] == 1 and meta[1] == 0:
            f.writelines('2') #landbird_background:water
            count2 = count2 + 1
            f.writelines('\n')
        elif meta[0] == 0 and meta[1] == 1:
            f.writelines('3') #waterbird_background:land
            f.writelines('\n')
            count3 = count3 + 1
        elif meta[0] == 1 and meta[1] == 1:
            f.writelines('4') #waterbird_background:water
            f.writelines('\n')
            count4 = count4 + 1
        else:
            raise NotImplementedError    
    f.close()

    Y_tr = torch.tensor(Y_tr)
    Y_te = torch.tensor(Y_te)
    X_tr = np.array(X_tr)
    X_te = np.array(X_te)
    return Data(X_tr, Y_tr, X_te, Y_te, handler, args_task)














