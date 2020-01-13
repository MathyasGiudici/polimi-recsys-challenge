from OwnUtils.Builder import Builder
from OwnUtils.Extractor import Extractor
from Utils.Base.NonPersonalizedRecommender import TopPop
import numpy as np
import pandas as pd


class FeatureAdder(object):

    def __init__(self, dataframe, group_length):
        self.builder = Builder()
        extractor = Extractor()

        self.dataframe = dataframe
        self.group_length = group_length
        self.urm = extractor.get_urm_all()
        self.icm = extractor.get_icm_all()
        self.users = extractor.get_target_users_of_recs()

        self.df_user_id_col = list(self.dataframe.loc[:, 'user_id'])
        self.df_item_id_col = list(self.dataframe.loc[:, 'item_id'])

        # Conversion from list of strings in list of int
        self.df_user_id_col = [int(i) for i in self.df_user_id_col]
        self.df_item_id_col = [int(i) for i in self.df_item_id_col]

    # USER PROFILE LENGTH FEATURE
    def add_user_profile_length(self):
        """
        Add the feature that count for each user the number of interactions inside the urm
        """
        print("Adding user profile length feature...")

        user_profile_len = np.ediff1d(self.urm.indptr)
        user_profile_len_col = []

        for user_id in self.df_user_id_col:
            user_profile_len_col.extend([user_profile_len[user_id]])

        self.dataframe['user_profile_len'] = pd.Series(user_profile_len_col, index=self.dataframe.index)
        print("Addition completed!")

    def add_user_age(self):
        return


    # POPULARITY OF EACH ITEM
    def add_item_popularity(self):
        """
        For each item add the feature of the number of times it has been bought, so related to its popularity
        """
        print("Adding TopPop items feature...")

        topPop = TopPop(self.urm)
        topPop.fit()

        topPop_score_list = []

        for user_id, item_id in zip(self.df_user_id_col, self.df_item_id_col):
            topPop_score = topPop._compute_item_score([user_id])[0, item_id]
            topPop_score_list.extend([int(topPop_score)])

        self.dataframe['item_popularity'] = pd.Series(topPop_score_list, index=self.dataframe.index)
        print("Addition completed!")