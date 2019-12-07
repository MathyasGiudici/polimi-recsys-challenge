from Hybrid.GeneralHybrid import GeneralHybrid

class RoundRobinHybrid(GeneralHybrid):

    def __init__(self,  urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, rr_order, withVariation):
        GeneralHybrid.__init__(self, urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw)

        self.hybrid_ratings = []
        self.rr_order = rr_order
        self.withVariation = withVariation


    def recommend(self, user, at=10):
        if self.withVariation:
            return self.recommend_variated(user, at)
        else:
             return self.recommend_basic(user, at)

    def recommend_basic(self, user, at=10):

        self.hybrid_ratings.clear()

        if self.p_icfknn is not None:
            self.hybrid_ratings.append(self.recommender_itemCFKNN.recommend(user, at=15))
        if self.p_ucfknn is not None:
            self.hybrid_ratings.append(self.recommender_userCFKNN.recommend(user, at=15))
        if self.p_cbfknn is not None:
            self.hybrid_ratings.append(self.recommender_itemCBFKNN.recommend(user, at=15))
        if self.p_slimbpr is not None:
            self.hybrid_ratings.append(self.recommender_slim_bpr.recommend(user, at=15))
        if self.p_puresvd is not None:
            self.hybrid_ratings.append(self.recommender_puresvd.recommend(user, at=15))
        if self.p_als is not None:
            self.hybrid_ratings.append(self.recommender_als.recommend(user, at=15))
        if self.p_cfw is not None:
            self.hybrid_ratings.append(self.recommender_cfw.recommend(user, at=15))

        recommended_items = []

        rank_position = 0
        while (rank_position < 15) and (len(recommended_items) < 10):
            for rec_order in self.rr_order:
                if self.hybrid_ratings[rec_order][rank_position] not in recommended_items:
                    recommended_items.append(self.hybrid_ratings[rec_order][rank_position])

            rank_position += 1

        if len(recommended_items)<10:
            raise EnvironmentError("ROUND_ROBIN: not enough recommendations!")

        return recommended_items[0:at]

    def recommend_variated(self, user, at=10):

        self.hybrid_ratings.clear()

        if self.p_icfknn is not None:
            self.hybrid_ratings.extend(self.recommender_itemCFKNN.recommend(user, at=10))
        if self.p_ucfknn is not None:
            self.hybrid_ratings.extend(self.recommender_userCFKNN.recommend(user, at=10))
        if self.p_cbfknn is not None:
            self.hybrid_ratings.extend(self.recommender_itemCBFKNN.recommend(user, at=10))
        if self.p_slimbpr is not None:
            self.hybrid_ratings.extend(self.recommender_slim_bpr.recommend(user, at=10))
        if self.p_puresvd is not None:
            self.hybrid_ratings.extend(self.recommender_puresvd.recommend(user, at=10))
        if self.p_als is not None:
            self.hybrid_ratings.extend(self.recommender_als.recommend(user, at=10))
        if self.p_cfw is not None:
            self.hybrid_ratings.extend(self.recommender_cfw.recommend(user, at=10))

        from collections import Counter
        recommended_dict = Counter(self.hybrid_ratings)
        recommended_items = list(recommended_dict.keys())

        return recommended_items[0:at]