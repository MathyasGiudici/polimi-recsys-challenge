import numpy as np

from OwnUtils.Extractor import Extractor

#from SLIM_BPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender

import Utils.Split.split_train_validation_leave_k_out as loo
from Utils.Split.DataReader_utils import remove_empty_rows_and_cols
from Utils.IR_feature_weighting import okapi_BM_25

class Hybrid(object):

    def __init__(self,  isTest , p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, weights):

        # Parameter saving
        self.isTest = isTest
        self.p_icfknn = p_icfknn
        self.p_ucfknn = p_ucfknn
        self.p_cbfknn = p_cbfknn
        self.p_slimbpr = p_slimbpr
        self.weights = weights

        # Getting matrices
        extractor = Extractor

        self.urm = extractor.get_interaction_matrix_all(extractor)
        self.urm_all = self.urm
       # self.urm = remove_empty_rows_and_cols(self.urm)[0]

        self.icm = extractor.get_icm_all(extractor)
        #self.icm = remove_empty_rows_and_cols(self.icm)[0]

        # Splitting into validation & testing in case of parameter tuning
        if isTest:
            matrices = loo.split_train_leave_k_out_user_wise(self.urm, 1, False, True)

            self.urm = matrices[0]
            self.urm_test = matrices[1]
            #self.urm_validation = matrices[2]

            # matrices = loo.split_train_leave_k_out_user_wise(self.icm, 1, True, True)
            #
            # self.icm = matrices[0]
            # self.icm_test = matrices[1]
            # self.icm_validation = matrices[2]

        self.icm_bm25 = self.icm.copy().astype(np.float32)
        self.icm_bm25 = okapi_BM_25(self.icm_bm25)
        self.icm_bm25 = self.icm_bm25.tocsr()

        # Creating recommenders
        self.recommender_itemCFKNN = ItemCFKNNRecommender(self.urm, self.icm)
        self.recommender_userCFKNN = UserCFKNNRecommender(self.urm)
        self.recommender_itemCBFKNN = ItemCBFKNNRecommender(self.urm_all, self.icm_bm25)


    def fit(self):
        self.recommender_itemCFKNN.fit(topK=self.p_icfknn["topK"],shrink=self.p_icfknn["shrink"])
        self.recommender_userCFKNN.fit(topK=self.p_ucfknn["topK"],shrink=self.p_ucfknn["shrink"])
        self.recommender_itemCBFKNN.fit(topK=self.p_cbfknn["topK"],shrink=self.p_cbfknn["shrink"])

    def recommend(self, user, at=10):
        self.hybrid_ratings = None

        self.hybrid_ratings = self.recommender_itemCFKNN.get_expected_ratings(user) * self.weights["icfknn"]
        self.hybrid_ratings += self.recommender_userCFKNN.get_expected_ratings(user) * self.weights["ucfknn"]
        self.hybrid_ratings += self.recommender_itemCBFKNN.get_expected_ratings(user) * self.weights["cbfknn"]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]