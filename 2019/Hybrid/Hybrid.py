import numpy as np

from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from MF.ALS import AlternatingLeastSquare
from MF.PureSVDRecommender import PureSVDRecommender
from CommonFeatureWeighting import CommonFeatureWeighting

from Utils.Base.IR_feature_weighting import okapi_BM_25


class Hybrid(object):

    def __init__(self,  urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, weights):

        # Parameter saving
        self.p_icfknn = p_icfknn
        self.p_ucfknn = p_ucfknn
        self.p_cbfknn = p_cbfknn
        self.p_slimbpr = p_slimbpr
        self.p_puresvd = p_puresvd
        self.p_als = p_als
        self.p_cfw = p_cfw

        self.weights = weights

        # Getting matrices
        self.urm = urm
        self.icm = icm

        self.icm_bm25 = self.icm.copy().astype(np.float32)
        self.icm_bm25 = okapi_BM_25(self.icm_bm25)
        self.icm_bm25 = self.icm_bm25.tocsr()

        # Creating recommenders
        if self.p_icfknn is not None:
            self.recommender_itemCFKNN = ItemCFKNNRecommender(self.urm.copy())
        if self.p_ucfknn is not None:
            self.recommender_userCFKNN = UserCFKNNRecommender(self.urm.copy())
        if self.p_cbfknn is not None:
            self.recommender_itemCBFKNN = ItemCBFKNNRecommender(self.urm.copy(), self.icm_bm25)
        if self.p_slimbpr is not None:
            self.recommender_slim_bpr = SLIM_BPR_Cython(self.urm.copy())
        if self.p_puresvd is not None:
            self.recommender_puresvd = PureSVDRecommender(self.urm.copy())
        if self.p_als is not None:
            self.recommender_als = AlternatingLeastSquare(self.urm.copy())

        self.hybrid_ratings = np.zeros(shape=(18495, 10))

    def fit(self):
        """
        Fit the different selected algorithms
        TODO: write the method in a more beautiful way
        """
        if self.p_icfknn is not None:
            self.recommender_itemCFKNN.fit(**self.p_icfknn)
        if self.p_ucfknn is not None:
            self.recommender_userCFKNN.fit(**self.p_ucfknn)
        if self.p_cbfknn is not None:
            self.recommender_itemCBFKNN.fit(**self.p_cbfknn)
        if self.p_slimbpr is not None:
            self.recommender_slim_bpr.fit(**self.p_slimbpr)
        if self.p_puresvd is not None:
            self.recommender_puresvd.fit(**self.p_puresvd)
        if self.p_als is not None:
            self.recommender_als.fit(**self.p_als)

        if self.p_cfw is not None:
            self.recommender_cfw = CommonFeatureWeighting(self.urm.copy(), self.icm_bm25, self.recommender_itemCFKNN.get_W_sparse())
            self.recommender_cfw.fit(**self.p_cfw)

    def recommend(self, user, at=10):

        if self.p_icfknn is not None:
            self.hybrid_ratings = self.recommender_itemCFKNN.get_expected_ratings(user) * self.weights["icfknn"]
        if self.p_ucfknn is not None:
            self.hybrid_ratings += self.recommender_userCFKNN.get_expected_ratings(user) * self.weights["ucfknn"]
        if self.p_cbfknn is not None:
            self.hybrid_ratings += self.recommender_itemCBFKNN.get_expected_ratings(user) * self.weights["cbfknn"]
        if self.p_slimbpr is not None:
            self.hybrid_ratings += self.recommender_slim_bpr.get_expected_ratings(user) * self.weights["slimbpr"]
        if self.p_puresvd is not None:
            self.hybrid_ratings += self.recommender_puresvd.get_expected_ratings(user) * self.weights["puresvd"]
        if self.p_als is not None:
            self.hybrid_ratings += self.recommender_als.get_expected_ratings(user) * self.weights["als"]

        if self.p_cfw is not None:
            self.hybrid_ratings += self.recommender_cfw.get_expected_ratings(user) * self.weights["cfw"]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]