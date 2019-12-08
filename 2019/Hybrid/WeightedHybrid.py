import numpy as np

from Hybrid.GeneralHybrid import GeneralHybrid

class WeightedHybrid(GeneralHybrid):

    def __init__(self, urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, p_p3a, p_rp3b, weights):
        GeneralHybrid.__init__(self, urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, p_p3a, p_rp3b)

        self.weights = weights

        n_user, n_items = urm.shape
        self.n_items = n_items

        self.hybrid_ratings = None

    def recommend(self, user, at=10):
        self.hybrid_ratings = np.zeros(self.n_items)

        # if self.p_icfknn is not None:
        #     self.hybrid_ratings = self.recommender_itemCFKNN.get_expected_ratings(user) * self.weights["icfknn"]
        # if self.p_ucfknn is not None:
        #     self.hybrid_ratings += self.recommender_userCFKNN.get_expected_ratings(user) * self.weights["ucfknn"]
        # if self.p_cbfknn is not None:
        #     self.hybrid_ratings += self.recommender_itemCBFKNN.get_expected_ratings(user) * self.weights["cbfknn"]
        # if self.p_slimbpr is not None:
        #     self.hybrid_ratings += self.recommender_slim_bpr.get_expected_ratings(user) * self.weights["slimbpr"]
        # if self.p_puresvd is not None:
        #     self.hybrid_ratings += self.recommender_puresvd.get_expected_ratings(user) * self.weights["puresvd"]
        # if self.p_als is not None:
        #     self.hybrid_ratings += self.recommender_als.get_expected_ratings(user) * self.weights["als"]
        #
        # if self.p_cfw is not None:
        #     self.hybrid_ratings += self.recommender_cfw.get_expected_ratings(user) * self.weights["cfw"]

        if self.p_icfknn is not None:
            self.hybrid_ratings += self.recommender_itemCFKNN.get_expected_ratings(user) * self.weights["icfknn"]
        if self.p_slimbpr is not None:
            self.hybrid_ratings += self.recommender_slim_bpr.get_expected_ratings(user) * self.weights["slimbpr"]
        if self.p_ucfknn is not None:
            self.hybrid_ratings += self.recommender_userCFKNN.get_expected_ratings(user) * self.weights["ucfknn"]
        if self.p_puresvd is not None:
            self.hybrid_ratings += self.recommender_puresvd.get_expected_ratings(user) * self.weights["puresvd"]
        if self.p_als is not None:
            self.hybrid_ratings += self.recommender_als.get_expected_ratings(user) * self.weights["als"]
        if self.p_cbfknn is not None:
            self.hybrid_ratings += self.recommender_itemCBFKNN.get_expected_ratings(user) * self.weights["cbfknn"]
        if self.p_p3a is not None:
            self.hybrid_ratings += self.recommender_p3a.get_expected_ratings(user) * self.weights["p3a"]
        if self.p_rp3b is not None:
            self.hybrid_ratings += self.recommender_rp3b.get_expected_ratings(user) * self.weights["rp3b"]

        if self.p_cfw is not None:
            self.hybrid_ratings += self.recommender_cfw.get_expected_ratings(user) * self.weights["cfw"]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.urm[user].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]