
from OwnUtils.Extractor import Extractor

from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from CFKNN.UserCFKNNRecommender import UserCFKNNRecommender
from CBFKNN.UserCFKNNRecommender import ItemCBFKNNRecommender

class Hybrid(object):

    def __init__(self, p_icfknn, p_ucfknn, p_cbfknn):
        self.urm = None
        self.icm = None

        self.recommender_itemCFKNN = ItemCFKNNRecommender(self.urm, self.icm)
        self.recommender_userCFKNN = UserCFKNNRecommender(self.urm, self.icm)
        self.recommender_itemCBFKNN = ItemCBFKNNRecommender(self.urm, self.icm)

        return

    def fit(self, urm):
        pass

    def recommend(self, user, at=10):
        pass