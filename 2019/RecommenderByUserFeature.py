import numpy as np
import scipy.sparse as sps
from OwnUtils.Extractor import Extractor
from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
import WeightConstants
from tqdm import tqdm
from Utils.Base.IR_feature_weighting import okapi_BM_25


class RecommenderByUserFeature(object):

    def __init__(self, urm_train, icm, urm_per_region_list, urm_per_age_list):
        self.urm_train = urm_train
        self.urm_per_region_list = urm_per_region_list
        self.urm_per_age_list = urm_per_age_list
        self.icm = icm

        self.icfknn_list = []
        self.ucfknn_list = []
        self.icbfknn_list =[]

        self.icm_bm25 = self.icm.copy().astype(np.float32)
        self.icm_bm25 = okapi_BM_25(self.icm_bm25)
        self.icm_bm25 = self.icm_bm25.tocsr()

        self.ratings = np.zeros(self.icm.shape[0])

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

        self.icfknn_list.append(ItemCFKNNRecommender(self.urm_train.copy()))
        self.ucfknn_list.append(UserCFKNNRecommender(self.urm_train.copy()))
        self.icbfknn_list.append(ItemCBFKNNRecommender(self.urm_train.copy(), self.icm_bm25.copy()))


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


    def recommend(self, user, at=10):
        for alg in self.icfknn_list:
            self.ratings += alg.get_expected_ratings(user)
        for alg in self.ucfknn_list:
            self.ratings += alg.get_expected_ratings(user)
        for alg in self.icbfknn_list:
            self.ratings += alg.get_expected_ratings(user)

        recommended_items = np.flip(np.argsort(self.ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm_train[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]