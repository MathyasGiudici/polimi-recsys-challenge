import numpy as np
from Hybrid.GeneralHybrid import GeneralHybrid

class RoundRobinHybrid(GeneralHybrid):

    def __init__(self,  urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, rr_order):
        GeneralHybrid.__init__(self, urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw)

        self.hybrid_ratings = []
        self.rr_order = rr_order


    def recommend(self, user, at=10):

        if self.p_icfknn is not None:
            self.hybrid_ratings.append(self.recommender_itemCFKNN.recommend(user, at=10))
        if self.p_ucfknn is not None:
            self.hybrid_ratings.append(self.recommender_userCFKNN.recommend(user, at=10))
        if self.p_cbfknn is not None:
            self.hybrid_ratings.append(self.recommender_itemCBFKNN.recommend(user, at=10))
        if self.p_slimbpr is not None:
            self.hybrid_ratings.append(self.recommender_slim_bpr.recommend(user, at=10))
        if self.p_puresvd is not None:
            self.hybrid_ratings.append(self.recommender_puresvd.recommend(user, at=10))
        if self.p_als is not None:
            self.hybrid_ratings.append(self.recommender_als.recommend(user, at=10))
        if self.p_cfw is not None:
            self.hybrid_ratings.append(self.recommender_cfw.recommend(user, at=10))

        recommended_items = []

        rank_position = 0
        while (rank_position < at) and (len(recommended_items) < 10):
            for rec_order in self.rr_order:
                recommended_items.append(self.hybrid_ratings[rec_order][rank_position])

            rank_position += 1

        return recommended_items[0:at]