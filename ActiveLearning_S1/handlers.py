import numpy as np
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class LINCS_Handler(Dataset):
    def __init__(self, X, Y, cell, balancesample=False, random_seed=6789):
        self.cell = cell
        self.random_seed = random_seed
        if balancesample:
            balanceIDXs = self.down_sampling(Y)
            X = X[balanceIDXs]
            Y = Y[balanceIDXs]
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
    
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
