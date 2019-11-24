import numpy as np

from OwnUtils.Extractor import Extractor

from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from SLIM_BPR.SLIM_BPR_Cython import SLIM_BPR_Cython

from Utils.Base.IR_feature_weighting import okapi_BM_25

class Hybrid(object):

    def __init__(self,  urm, icm , p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, weights):

        # Parameter saving
        self.p_icfknn = p_icfknn
        self.p_ucfknn = p_ucfknn
        self.p_cbfknn = p_cbfknn
        self.p_slimbpr = p_slimbpr
        self.weights = weights

        # Getting matrices
        self.urm = urm
        self.icm = icm


        self.icm_bm25 = self.icm.copy().astype(np.float32)
        self.icm_bm25 = okapi_BM_25(self.icm_bm25)
        self.icm_bm25 = self.icm_bm25.tocsr()

        # Creating recommenders
        self.recommender_itemCFKNN = ItemCFKNNRecommender(self.urm, self.icm)
        self.recommender_userCFKNN = UserCFKNNRecommender(self.urm)
        self.recommender_itemCBFKNN = ItemCBFKNNRecommender(self.urm, self.icm_bm25)
        self.recommender_slim_bpr = SLIM_BPR_Cython(self.urm)


    def fit(self):
        self.recommender_itemCFKNN.fit(topK=self.p_icfknn["topK"],shrink=self.p_icfknn["shrink"])
        self.recommender_userCFKNN.fit(topK=self.p_ucfknn["topK"],shrink=self.p_ucfknn["shrink"])
        self.recommender_itemCBFKNN.fit(topK=self.p_cbfknn["topK"],shrink=self.p_cbfknn["shrink"])
        self.recommender_slim_bpr.fit(epochs=self.p_slimbpr["epochs"])

    def recommend(self, user, at=10):
        self.hybrid_ratings = None

        self.hybrid_ratings = self.recommender_itemCFKNN.get_expected_ratings(user) * self.weights["icfknn"]
        self.hybrid_ratings += self.recommender_userCFKNN.get_expected_ratings(user) * self.weights["ucfknn"]
        self.hybrid_ratings += self.recommender_itemCBFKNN.get_expected_ratings(user) * self.weights["cbfknn"]
        self.hybrid_ratings += self.recommender_slim_bpr.get_expected_ratings(user) * self.weights["slimbpr"]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]