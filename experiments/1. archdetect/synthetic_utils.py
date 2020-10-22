import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from sklearn import metrics


class synth_model:
    def __init__(self, test_id, input_value, base_value):
        self.test_id = test_id
        self.input_value = input_value
        self.base_value = base_value

    def and_func(self, X, inter):
        bool_cols = []
        indices = []
        for i, val in inter:
            bool_cols.append(X[:, i] == val)
            indices.append(i)
        bool_out = np.all(np.array(bool_cols), axis=0)
        gt = list(itertools.combinations(indices, 2))
        return np.where(bool_out, 1, -1), gt

    def preprocess(self, X):
        Y = np.zeros(X.shape[0])
        Y += X.sum(1)
        p = X.shape[1]
        q = p // 4
        gts = []
        return Y, p, q, gts

    def synth0(self, X):
        Y, p, q, gts = self.preprocess(X)  # simple sum, no interactions
        return Y, gts

    def synth1(self, X):
        Y, p, q, gts = self.preprocess(X)
        #         Y += X.sum(1)
        gts = []
        for i in range(q):
            for j in range(q):
                Y += X[:, i] * X[:, j]
                gts.append((i, j))
        for i in range(q, q * 2):
            for j in range(q * 2, q * 3):
                Y += X[:, i] * X[:, j]
                gts.append((i, j))
        return Y, gts

    def synth2(self, X):
        Y, p, q, gts = self.preprocess(X)
        #         Y += X.sum(1)
        Y1, gt1 = self.and_func(X, [(i, self.input_value) for i in range(q * 2)])
        Y2, gt2 = self.and_func(X, [(i, self.input_value) for i in range(q, q * 3)])
        Y += Y1 + Y2
        gts += gt1 + gt2
        return Y, gts

    def synth3(self, X):
        Y, p, q, gts = self.preprocess(X)
        #         Y += X.sum(1)

        Y1, gt1 = self.and_func(X, [(i, self.base_value) for i in range(q * 2)])
        Y2, gt2 = self.and_func(X, [(i, self.input_value) for i in range(q, q * 3)])
        Y += Y1 + Y2
        gts += gt1 + gt2
        return Y, gts

    def synth4(self, X):
        Y, p, q, gts = self.preprocess(X)
        #         Y += X.sum(1)

        range1 = [(i, self.input_value) for i in range(2)]
        range2 = [(i, self.base_value) for i in range(2, 3)]
        Y1, gt1 = self.and_func(X, range1 + range2)
        Y2, gt2 = self.and_func(X, [(i, self.input_value) for i in range(q, q * 3)])
        Y += Y1 + Y2
        gts += gt1 + gt2
        return Y, gts

    def synth(self, X):
        if self.test_id == 0:
            Y, gts = self.synth0(X)
        elif self.test_id == 1:
            Y, gts = self.synth1(X)
        elif self.test_id == 2:
            Y, gts = self.synth2(X)
        elif self.test_id == 3:
            Y, gts = self.synth3(X)
        elif self.test_id == 4:
            Y, gts = self.synth4(X)
        else:
            raise ValueError
        return Y, gts

    def get_gts(self, num_features):
        X = np.ones((1, num_features)) * self.input_value
        _, gts = self.synth(X)
        return gts

    def __call__(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X, 0)
        Y, _ = self.synth(X)
        return np.expand_dims(Y, 1)


def get_auc(inter_scores, gts):
    gt_vec = []
    pred_vec = []
    for inter in inter_scores:
        #     print(inter[0])
        #     print(inter)
        pred_vec.append(inter[1])
        if inter[0] in gts:
            gt_vec.append(1)
        else:
            gt_vec.append(0)

    fpr, tpr, thresholds = metrics.roc_curve(gt_vec, pred_vec, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def gen_data_samples(model, input_value, base_value, p, n=30000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = []
    for i in range(n):
        X.append(np.random.choice([input_value, base_value], p))
    X = np.stack(X)

    Y = model(X).squeeze()
    return X, Y
