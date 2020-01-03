#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Utils.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Utils.Base.Recommender_utils import check_matrix

import numpy as np
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps


class PureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, URM_train):
        super(PureSVDRecommender, self).__init__(URM_train)


    def fit(self, num_factors=100, random_seed = None):

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      #n_iter=5,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")

    def get_expected_ratings(self, user_id):
        if np.isscalar(user_id):
            user_id_array = np.atleast_1d(user_id)
        else:
            user_id_array = user_id

        return np.squeeze(self._compute_item_score(user_id_array))

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM_train[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

