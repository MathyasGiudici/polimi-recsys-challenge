import numpy as np
import scipy.sparse as sps
from OwnUtils.Extractor import Extractor
from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from MF.PureSVDRecommender import PureSVDRecommender
from SLIM.SLIM_BPR_Cython import SLIM_BPR_Cython
import WeightConstants
from tqdm import tqdm
from Utils.Base.IR_feature_weighting import okapi_BM_25


class RecommenderByUserFeature(object):

    def __init__(self, urm_train, icm, urm_per_region_list, urm_per_age_list, weights, add_pure_svd=False, add_slim_bpr=False):
        self.urm_train = urm_train
        self.urm_per_region_list = urm_per_region_list
        self.urm_per_age_list = urm_per_age_list
        self.icm = icm

        self.add_pure_svd = add_pure_svd
        self.add_slim_bpr = add_slim_bpr

        self.weights = weights

        self.icfknn_list = []
        self.ucfknn_list = []
        self.icbfknn_list =[]

        self.icm_bm25 = self.icm.copy().astype(np.float32)
        self.icm_bm25 = okapi_BM_25(self.icm_bm25)
        self.icm_bm25 = self.icm_bm25.tocsr()

        self.ratings = None

        self.extractor = Extractor()

        # Creation of the list of algortms that have to be used
        if self.urm_per_region_list is not None:
            for urm in self.urm_per_region_list:
                sps.csr_matrix(urm)
                self.icfknn_list.append(ItemCFKNNRecommender(urm.copy()))
                self.ucfknn_list.append(UserCFKNNRecommender(urm.copy()))
                self.icbfknn_list.append(ItemCBFKNNRecommender(urm.copy(), self.icm_bm25.copy()))

        if self.urm_per_age_list is not None:
            for urm in self.urm_per_age_list:
                sps.csr_matrix(urm)
                self.icfknn_list.append(ItemCFKNNRecommender(urm.copy()))
                self.ucfknn_list.append(UserCFKNNRecommender(urm.copy()))
                self.icbfknn_list.append(ItemCBFKNNRecommender(urm.copy(), self.icm_bm25.copy()))

        # self.icfknn_list.append(ItemCFKNNRecommender(self.urm_train.copy()))
        # self.ucfknn_list.append(UserCFKNNRecommender(self.urm_train.copy()))
        # self.icbfknn_list.append(ItemCBFKNNRecommender(self.urm_train.copy(), self.icm_bm25.copy()))

        if self.add_pure_svd:
            self.pure_SVD = PureSVDRecommender(self.urm_train.copy())
        if self.add_slim_bpr:
            self.slim_bpr = SLIM_BPR_Cython(self.urm_train.copy())


    def fit(self):
        print("Computing the similarity matrices for the itemCFKNN...")
        for alg in self.icfknn_list:
            alg.fit(**WeightConstants.ICFKNN)
        print("Computation completed!")

        print("Computing the similarity matrices for the userCFKNN...")
        for alg in self.ucfknn_list:
            alg.fit(**WeightConstants.UCFKNN)
        print("Computation completed!")

        print("Computing the similarity matrices for the itemCBFKNN...")
        for alg in self.icbfknn_list:
            alg.fit(**WeightConstants.CBFKNN)
        print("Computation completed!")

        if self.add_pure_svd:
            self.pure_SVD.fit(**WeightConstants.PURE_SVD)
        if self.add_slim_bpr:
            self.slim_bpr.fit(**WeightConstants.SLIM_BPR)


    def recommend(self, user, at=10):
        self.ratings = np.zeros(self.icm.shape[0])

        for alg in self.icfknn_list:
            self.ratings += alg.get_expected_ratings(user) * self.weights["icfknn"]
        for alg in self.ucfknn_list:
            self.ratings += alg.get_expected_ratings(user) * self.weights["ucfknn"]
        for alg in self.icbfknn_list:
            self.ratings += alg.get_expected_ratings(user) * self.weights["cbfknn"]

        if self.add_pure_svd:
            self.ratings += self.pure_SVD.get_expected_ratings(user) * self.weights["puresvd"]
        if self.add_slim_bpr:
            self.ratings += self.slim_bpr.get_expected_ratings(user) * self.weights["slimbpr"]

        recommended_items = np.flip(np.argsort(self.ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm_train[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]