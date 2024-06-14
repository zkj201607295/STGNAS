import numpy as np
import torch
import scipy.sparse as sp
from utils import data_loader, sparse_mx_to_torch_sparse_tensor


class Sampler:
    def __init__(self, dataset, data_path='data', task_type="semi"):
        self.dataset = dataset
        self.data_path = data_path
        (self.adj,
         self.train_adj,
         self.features,
         self.train_features,
         self.labels,
         self.idx_train,
         self.idx_val,
         self.idx_test,
         self.degree) = data_loader(dataset, data_path, False, task_type)

        self.features = torch.FloatTensor(self.features).float()
        self.train_features = torch.FloatTensor(self.train_features).float()

        self.labels_torch = torch.LongTensor(self.labels)
        self.idx_train_torch = torch.LongTensor(self.idx_train)
        self.idx_val_torch = torch.LongTensor(self.idx_val)
        self.idx_test_torch = torch.LongTensor(self.idx_test)

        self.pos_train_idx = np.where(self.labels[self.idx_train] == 1)[0]
        self.neg_train_idx = np.where(self.labels[self.idx_train] == 0)[0]

        self.nfeat = self.features.shape[1]
        self.nclass = int(self.labels.max().item() + 1)
        self.trainadj_cache = {}
        self.adj_cache = {}
        self.degree_p = None
        self.cuda = True if torch.cuda.is_available() else False

    def _preprocess_adj(self, adj):
        r_adj = self.aug_normalized_adjacency(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
        if self.cuda:
            r_adj = r_adj.cuda()
        return r_adj

    def _preprocess_fea(self, fea):
        if self.cuda:
            return fea.cuda()
        else:
            return fea

    def stub_sampler(self, norm="Norm"):
        if norm in self.trainadj_cache:
            r_adj = self.trainadj_cache[norm]
        else:
            r_adj = self._preprocess_adj(self.train_adj)
            self.trainadj_cache[norm] = r_adj
        fea = self._preprocess_fea(self.train_features)
        return r_adj, fea

    def aug_normalized_adjacency(self, adj):
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def get_test_set(self, normalization="AugNormAdj"):
        return self.stub_sampler(normalization)

    def get_val_set(self, normalization="AugNormAdj"):
        return self.stub_sampler(normalization)

    def get_label_and_idxes(self, cuda=False):
        if cuda:
            return self.labels_torch.cuda(), self.idx_train_torch.cuda(), self.idx_val_torch.cuda(), self.idx_test_torch.cuda()
        return self.labels_torch, self.idx_train_torch, self.idx_val_torch, self.idx_test_torch
