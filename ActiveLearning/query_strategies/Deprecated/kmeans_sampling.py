import numpy as np
import torch
from .strategy import jointStrategy
from sklearn.cluster import KMeans

class KMeansSampling(jointStrategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(KMeansSampling, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # embeddings = self.get_embeddings(unlabeled_data).numpy()
        unlabeled_idxs, unlabeled_drugs = self.dataset.get_unlabeled_drugs()
        embeddings = self.get_embeddings()
        embeddings = torch.concatenate(embeddings, 1).numpy()

        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
        
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        return unlabeled_idxs[q_idxs]
