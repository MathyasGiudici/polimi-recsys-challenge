import scipy.sparse as sps
import numpy as np
from OwnUtils.Extractor import Extractor

POPULARITY_THRESHOLD = 50
USER_RATIO_THRESHOLD = 0.5
LONGTAIL_WEIGHT = 0.7


class PostProcessing(object):

    def __init__(self):
        ex = Extractor()
        self.urm = ex.get_urm_all()
        self.icm = ex.get_icm_all()

        self.short_head = []
        self.long_tail = []

        self.user_short_head_ratio = 0
        self.build_item_popularity_list()

    # GET USER PREFERENCE BETWEEN SHORT HEAD AND LONG TAIL
    def get_user_category_pref(self):
        if self.user_short_head_ratio > USER_RATIO_THRESHOLD:
            return self.user_short_head_ratio
        else:
            return 1 - self.user_short_head_ratio

    def build_item_popularity_list(self):
        urm_coo = self.urm.tocoo()
        cols = []
        for i in urm_coo.col:
            cols.append(i)

        item_occurrencies = list(np.bincount(np.array(cols)))

        # sorted_items = sorted(item_occurrencies, reverse=True)
        # print(sorted_items)

        for index in range(0, len(item_occurrencies)):
            if item_occurrencies[index] > POPULARITY_THRESHOLD:
                self.short_head.append(index)
            else:
                self.long_tail.append(index)

    # RETURN THE TYPOLOGY OF USER, IF HE PREFERS POPULAR OR UNPOPULAR ITEMS
    def is_long_tail_user(self, user, train):
        user_profile = train[user].tocoo()
        user_items = []
        for i in user_profile.col:
            user_items.append(i)

        count_pop = 0

        for i in user_items:
            if i in self.short_head:
                count_pop += 1

        self.user_short_head_ratio = count_pop / len(user_items)


    def rerank_scores(self, items_base_scores: list):

        R_list = []

        for base_score in items_base_scores:
            weighted_base_score = (1 - LONGTAIL_WEIGHT) * base_score


            # weighted_category_booster = LONGTAIL_WEIGHT * self.user_short_head_ratio * (1 - )
        return


if __name__ == '__main__':
    pp = PostProcessing()
    ex = Extractor()
    pp.is_long_tail_user(0, ex.get_urm_all())
