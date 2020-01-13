from Hybrid.WeightedHybrid import WeightedHybrid
from OwnUtils.Extractor import Extractor
from OwnUtils.Builder import Builder
from OwnUtils.Writer import Writer
import ParametersTuning
from XGBoostDataframe import XGBoostDataframe
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm
import WeightConstants
from Utils.Base.NonPersonalizedRecommender import TopPop

import random
import Utils.Split.split_train_validation_leave_k_out as loo

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np

"""
Specify the report and the submission in which we will write the results
"""
report_counter = 10
submission_counter = 2

params = {
            'max_depth': 3,  # the maximum depth of each tree
            'n_estimators': 300,
            'learning_rate': 0.05,
            'eta': 0.3,  # step for each iteration
            'silent': 1,  # keep it quiet
            'num_class': 3,
            'objective': 'rank:pairwise',  # error evaluation for multiclass training
            'eval_metric': 'map'}  # evaluation metric


class XGBoost(object):

    def __init__(self, cutoff, cbfknn=False, icfknn=False, ucfknn=False, slim_bpr=False, pure_svd=False, als=False,
                 cfw=False, p3a=False, rp3b=False, slim_en=False):
        """
        Initialization of the generic runner in which we decide whether or not use an algorithm
        """
        self.cutoff = cutoff
        self.cbfknn = cbfknn
        self.icfknn = icfknn
        self.ucfknn = ucfknn
        self.slim_bpr = slim_bpr
        self.pure_svd = pure_svd
        self.als = als
        self.cfw = cfw
        self.p3a = p3a
        self.rp3b = rp3b
        self.slim_en = slim_en

        self.writer = Writer
        self.extractor = Extractor()
        self.df_builder = XGBoostDataframe(self.cutoff)
        self.result_dict = None

        self.urm_train = None
        self.urm_validation = None
        self.icm = self.extractor.get_icm_all()

        self.p_cbfknn = None
        self.p_icfknn = None
        self.p_ucfknn = None
        self.p_slimbpr = None
        self.p_puresvd = None
        self.p_als = None
        self.p_cfw = None
        self.p_p3a = None
        self.p_rp3b = None
        self.p_slimen = None

        self.target_users = []
        self.results = []

        self.df_user_id_col = []
        self.df_item_id_col = []

        self.df_train = pd.DataFrame
        self.df_test = pd.DataFrame

    def run(self, is_test):
        """
        From here we start each algorithm.
        :param is_test: specifies if we want to write a report or a submission
        """
        if is_test:

            # CREATION OF THE VALIDATIONS FOR EACH PART OF THE TRAIN
            vals = []
            urms = []
            target_profiles = []

            for i in range(1, 5):
                urm_to_predict = self.extractor.get_single_urm(i)

                matrices = loo.split_train_leave_k_out_user_wise(urm_to_predict, 1, False, True)

                target_users_profile = matrices[0]
                target_profiles.append(target_users_profile)

                val = matrices[1]
                vals.append(val)

                urm = self.extractor.get_others_urm_vstack(i)

                urms.append(urm)

            if self.icfknn:
                self.p_icfknn = ParametersTuning.ICFKNN_BEST
            if self.cbfknn:
                self.p_cbfknn = ParametersTuning.CBFKNN_BEST
            if self.rp3b:
                self.p_rp3b = ParametersTuning.RP3B_BEST
            if self.slim_en:
                self.p_slimen = ParametersTuning.SLIM_ELASTIC_NET_BEST

            # URM splitted in 4 smaller URMs for cross-validation
            for i in range(0, 4):
                self.urm_validation = vals[i].copy()
                self.urm_train = urms[i].copy()
                self.target_users = self.extractor.get_target_users_of_specific_part(i + 1)

                # GETTING THE RECOMMENDATIONS FOR THE TRAIN DATAFRAME
                user_ids, item_ids = self.evaluate(i + 1, target_profiles[i])
                self.df_user_id_col.extend(user_ids)
                self.df_item_id_col.extend(item_ids)

            # print(self.df_user_id_col[0:100])
            # print(self.df_item_id_col[0:100])
            # print(len(self.df_user_id_col))
            # print(len(self.df_item_id_col))

            self.score_ranking()

    def write_submission(self, users):
        pass

    # RE-RANKING AND EVALUATION
    def evaluate(self, index: int, target_users_profile):
        """
        This method capture the predictions of the CrossValidation running the Hybrid
        :param index: number of iteration (from 1 to 4) depending on the current sub-URM
        :param target_users_profile: profile of the users wanted to predict
        :return:
        """
        weight = {'icfknn': 1, 'ucfknn': 1, 'cbfknn': 1, 'slimbpr': 1, 'puresvd': 1, 'als': 1, 'cfw': 1, 'p3a': 1,
                  'rp3b': 1, 'slimen': 1}

        recommender = WeightedHybrid(self.urm_train, self.icm, self.p_icfknn, self.p_ucfknn, self.p_cbfknn,
                                     self.p_slimbpr, self.p_puresvd, self.p_als, self.p_cfw, self.p_p3a,
                                     self.p_rp3b, self.p_slimen, weight, seen_items=target_users_profile)
        recommender.fit()

        user_ids = []
        item_ids = []

        # SELECTING THE BEST RECOMMENDATIONS
        for n_user in self.target_users:
            recommendations = recommender.recommend(n_user, at=self.cutoff)
            user_ids.extend([n_user] * len(recommendations))
            item_ids.extend(recommendations)

        return user_ids, item_ids


    # RE-RANKING OF THE SCORES WITH XGBOOST
    def score_ranking(self):
        """
        Preparation of dataframes and use of XGBoost
        :return:
        """
        # CREATING DATAFRAMES FOR XGBOOST
        print(">>> Preparing the two DataFrames...")
        # Train dataframe
        self.df_train = self.df_builder.build_base_dataframe(users=self.df_user_id_col, items=self.df_item_id_col)
        self.df_builder.build_whole_dataframe(self.df_train)

        # Test dataframe
        self.df_test = self.df_builder.retrieve_test_dataframe()

        # BUILD TRAIN AND TEST GROUPS
        train_group = []
        test_group = []

        train_user_ids = list(self.df_train.loc[:, 'user_id'].values)
        test_user_ids = list(self.df_test.loc[:, 'user_id'].values)

        train_group.extend([self.cutoff] * len(set(train_user_ids)))
        test_group.extend([self.cutoff] * len(set(test_user_ids)))

        # DROP USELESS COLUMNS OF DF_TRAIN AND DF_TEST
        train_dropped = self.df_train.drop(labels={'user_id', 'item_id'}, axis=1)
        test_dropped = self.df_test.drop(labels={'user_id', 'item_id'}, axis=1)

        print(">>> DataFrames well formed and ready to be used!")

        # LGBM TO TRAIN FASTER ON GPU
        # lgbm_group = self.xgb_dataframe.groupby('user_id').size().values

        # lightGBM_ranker = lgb.LGBMRanker(device='gpu')
        # lightGBM_ranker.fit(train_dropped, users, lgbm_group)

        # XGB RANKER AT WORK
        print(">>> Fitting of the XGB model...")
        xbg_ranker = xgb.XGBRanker()
        xbg_ranker.fit(train_dropped, train_user_ids, train_group)
        print(">>> Fitting completed!")
        # model = xgb.train(params,
        #                   dtrain,
        #                   num_round,
        #                   verbose_eval=2,
        #                   early_stopping_rounds=20)

        # print(xgb_regressor.predict(X_test))

        print(">>> Predicting the scores...")
        predictions = xbg_ranker.predict(test_dropped)
        print(">>> DONE")

        print(predictions[0:100])
        # print("user array:" + str(len(self.user_recommendations_user_id)))
        # print(" prediction array:" + str(len(model.predict())))


    def add_top_pop_items(self):
        # BUILDING POPULARITY ITEMS
        print("Adding TopPop items feature...")

        topPop = TopPop(self.urm_train)
        topPop.fit()

        topPop_score_list = []

        for user_id, item_id in zip(self.user_recommendations_user_id, self.user_recommendations_items):
            topPop_score = topPop._compute_item_score([user_id])[0, item_id]
            topPop_score_list.append(topPop_score)

        self.xgb_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=self.xgb_dataframe.index)
        print("Addition completed!")


    def add_user_profile_length(self):
        # BUILDING USER PROFILE LENGTH
        print("Adding user profile length feature...")

        user_profile_len = np.ediff1d(self.urm_train.indptr)

        user_profile_len_list = []

        for user_id in self.user_recommendations_user_id:
            user_profile_len_list.append(user_profile_len[user_id])

        self.xgb_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=self.xgb_dataframe.index)
        print("Addition completed!")

    def add_item_asset(self):
        # BUILDING ITEM ASSET
        print("Adding item asset feature...")

        icm_asset_df = self.builder.build_icm_asset_dataframe()

        assets = []

        j = 0
        for i in range(0, self.icm.shape[0]):
            if icm_asset_df.iloc[j]['row'] == i:
                assets.append(icm_asset_df.iloc[j]['data'])
                j += 1
            else:
                assets.append(0)

        icm_asset_list = []

        for item_id in self.user_recommendations_items:
            icm_asset_list.append(assets[item_id])

        self.xgb_dataframe['item_asset'] = pd.Series(icm_asset_list, index=self.xgb_dataframe.index)
        print("Addition completed!")

    def add_item_price(self):
        # BUILDING ITEM PRICE
        print("Adding item price feature...")

        icm_price_df = self.builder.build_icm_price_dataframe()

        prices = []

        j = 0
        for i in range(0, self.icm.shape[0]):
            if icm_price_df.iloc[j]['row'] == i:
                prices.append(icm_price_df.iloc[j]['data'])
                j += 1
            else:
                prices.append(0)

        icm_asset_list = []

        for item_id in self.user_recommendations_items:
            icm_asset_list.append(prices[item_id])

        self.xgb_dataframe['item_price'] = pd.Series(icm_asset_list, index=self.xgb_dataframe.index)
        print("Addition completed!")

    def add_item_subclass(self):
        # BUILDING ITEM SUBCLASS
        print("Adding item subclass feature...")

        icm_subclass_df = self.builder.build_icm_subclass_dataframe()

        subclasses = []

        j = 0
        for i in range(0, self.icm.shape[0]):
            if icm_subclass_df.iloc[j]['row'] == i:
                subclasses.append(icm_subclass_df.iloc[j]['col'])
                j += 1
            else:
                subclasses.append(0)

        icm_asset_list = []

        for item_id in self.user_recommendations_items:
            icm_asset_list.append(subclasses[item_id])

        self.xgb_dataframe['item_subclass'] = pd.Series(icm_asset_list, index=self.xgb_dataframe.index)
        print("Addition completed!")


if __name__ == "__main__":
    algorithms_choice = {
        "icfknn": True,
        "ucfknn": False,
        "cbfknn": False,
        "slim_bpr": False,
        "pure_svd": False,
        "als": False,
        "cfw": False,
        "p3a": False,
        "rp3b": False,
        "slim_en": False,
    }

    is_test = True
    at = 20

    runner = XGBoost(at, **algorithms_choice)
    runner.run(is_test)