#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""

import numpy as np
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
import multiprocessing
from multiprocessing import Pool
from functools import partial


class SLIMElasticNetRecommender(object):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, URM_train, target_users_profile):

        self.URM_train = URM_train
        self.target_users_profile = target_users_profile
        self.analyzed_items = 0

    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=self.alpha,
                           l1_ratio=self.l1_ratio,
                           positive=self.positive_only,
                           fit_intercept=self.fit_intercept,
                           copy_X=self.copy_X,
                           precompute=self.precompute,
                           selection=self.selection,
                           max_iter=self.max_iter,
                           tol=self.tol)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        # nnz_idx = model.coef_ > 0.0

        local_topK = min(len(model.sparse_coef_.data) - 1, self.topK)

        relevant_items_partition = (-model.coef_).argpartition(local_topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, alpha=1e-4, l1_ratio=0.1, fit_intercept=False, copy_X=False, precompute=False, selection='random',
            max_iter=20, tol=1e-4, topK=100, positive_only=True, workers=multiprocessing.cpu_count()):

        assert l1_ratio >= 0 and l1_ratio <= 1, "SLIM_ElasticNet: l1_ratio must be between 0 and 1, provided value " \
                                                "was {}".format(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = max_iter
        self.tol = tol
        self.topK = topK
        self.positive_only = positive_only
        self.workers = workers

        self.URM_train.tocsc()
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel

        print("Iterating for " + str(n_items) + "times...")

        # oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train.copy(), topK=self.topK)

        # creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)

        # avvio il pool passando la funzione (con la parte fissa dell'input)
        # e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
        self.recs = self.target_users_profile.dot(self.W_sparse)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.recs[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

