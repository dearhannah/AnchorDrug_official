import numpy as np
import torch
from .strategy import jointStrategy
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class KCenterGreedy(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KCenterGreedy, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        labeled_idxs, train_drugs = self.dataset.get_train_drugs()
        embeddings = self.get_embeddings(type='all')
        embeddings = torch.concatenate(embeddings, 1)
        embeddings = embeddings.reshape(len(train_drugs),-1).numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        row, col = np.diag_indices_from(dist_mat)
        dist_mat[row, col] = np.inf

        if sum(labeled_idxs) == 0:
            mat_min = dist_mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[q_idx_]
            labeled_idxs[q_idx] = True
            n = n-1

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]
        for i in range(n):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_drug_idxs ^ labeled_idxs)]