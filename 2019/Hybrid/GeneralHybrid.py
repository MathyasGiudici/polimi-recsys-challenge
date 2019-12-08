import numpy as np

from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
from MF.ALS import AlternatingLeastSquare
from MF.PureSVDRecommender import PureSVDRecommender
from Graph.P3A import P3alphaRecommender
from Graph.RP3B import RP3betaRecommender
from CommonFeatureWeighting import CommonFeatureWeighting

from Utils.Base.IR_feature_weighting import okapi_BM_25


class GeneralHybrid(object):

    def __init__(self,  urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, p_p3a, p_rp3b):

        # Parameter saving
        self.p_icfknn = p_icfknn
        self.p_ucfknn = p_ucfknn
        self.p_cbfknn = p_cbfknn
        self.p_slimbpr = p_slimbpr
        self.p_puresvd = p_puresvd
        self.p_als = p_als
        self.p_cfw = p_cfw
        self.p_p3a = p_p3a
        self.p_rp3b = p_rp3b

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
        if self.p_p3a is not None:
            self.recommender_p3a = P3alphaRecommender(self.urm.copy())
        if self.p_rp3b is not None:
            self.recommender_rp3b = RP3betaRecommender(self.urm.copy())

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

        if self.p_p3a is not None:
            self.recommender_p3a.fit(**self.p_p3a)
        if self.p_rp3b is not None:
            self.recommender_rp3b.fit(**self.p_rp3b)

    def recommend(self, user, at=10):

        raise NotImplementedError(
            "BaseRecommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")