from Hybrid.WeightedHybrid import WeightedHybrid
from OwnUtils.Extractor import Extractor
from OwnUtils.Builder import Builder
from OwnUtils.Writer import Writer
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm
import WeightConstants
from Utils.Base.NonPersonalizedRecommender import TopPop

import random
import Utils.Split.split_train_validation_leave_k_out as loo

from sklearn.model_selection import train_test_split
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

    def __init__(self, cbfknn=True, icfknn=True, ucfknn=True, slim_bpr=True, pure_svd=True, als=True, cfw=True):
        """
        Initialization of the generic runner in which we decide whether or not use an algorithm
        """
        self.cbfknn = cbfknn
        self.icfknn = icfknn
        self.ucfknn = ucfknn
        self.slim_bpr = slim_bpr
        self.pure_svd = pure_svd
        self.als = als
        self.cfw = cfw

        self.is_test = None
        self.writer = Writer
        self.result_dict = None

        self.urm_train = None
        self.urm_validation = None
        self.urm_test = None
        self.urm_post_validation = None
        self.icm = None

        self.p_cbfknn = None
        self.p_icfknn = None
        self.p_ucfknn = None
        self.p_slimbpr = None
        self.p_puresvd = None
        self.p_als = None
        self.p_cfw = None

        if self.cbfknn:
            self.p_cbfknn = WeightConstants.CBFKNN
        if self.icfknn:
            self.p_icfknn = WeightConstants.ICFKNN
        if self.ucfknn:
            self.p_ucfknn = WeightConstants.UCFKNN
        if self.slim_bpr:
            self.p_slimbpr = WeightConstants.SLIM_BPR
        if self.pure_svd:
            self.p_puresvd = WeightConstants.PURE_SVD
        if self.als:
            self.p_als = WeightConstants.ALS
        if self.cfw:
            self.p_cfw = WeightConstants.CFW

        self.user_recommendations_user_id = []
        self.user_recommendations_items = []
        self.cutoff = 20
        self.xgb_dataframe = None

        self.builder = Builder()
        self.users = []

        self.train_users = []
        self.test_users = []

    def run(self, is_test):
        """
        From here we start each algorithm.
        :param is_test: specifies if we want to write a report or a submission
        """
        self.is_test = is_test

        if self.is_test:
            extractor = Extractor()
            urm = extractor.get_urm_all()
            self.icm = extractor.get_icm_all()
            # self.icm_dirty = extractor.get_icm_price_dirty()

            # Splitting into post-validation & testing in case of parameter tuning
            matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

            self.urm_post_validation = matrices[0]
            self.urm_test = matrices[1]

            # ONLY TRAIN AND TEST
            self.urm_train = self.urm_post_validation

            ######################################################################

            # Splitting the post-validation matrix in train & validation
            # (Problem of merging train and validation again at the end => loo twice)
            # matrices_for_validation = loo.split_train_leave_k_out_user_wise(self.urm_post_validation, 1, False, True)
            # self.urm_train = matrices_for_validation[0]
            # self.urm_validation = matrices_for_validation[1]

            self.evaluate()

        else:
            extractor = Extractor()
            users = extractor.get_target_users_of_recs()
            self.urm_train = extractor.get_urm_all()
            #self.icm = extractor.get_icm_all()

            self.write_submission(users)

    def write_submission(self, users):
        pass

    def evaluate(self):
        weight = {"icfknn": 1}

        recommender = WeightedHybrid(self.urm_train, self.icm, self.p_icfknn, None, None,
                                        None, None, None, None, None, None, weight)
        recommender.fit()

        # SELECTING BEST 20 RECOMMENDATION
        for n_user in range(0, self.urm_test.shape[0]):
            recommendations = recommender.recommend(n_user, at=self.cutoff)

            self.user_recommendations_user_id.extend([n_user] * len(recommendations))
            self.user_recommendations_items.extend(recommendations)

        # CREATING THE DATAFRAME FOR XGBOOST
        self.xgb_dataframe = pd.DataFrame({"user_id": self.user_recommendations_user_id, "item_id": self.user_recommendations_items})

        ############################
        ###### ADDING FEATURES #####
        ############################

        # BUILDING POPULARITY ITEMS
        # self.add_top_pop_items()

        # BUILDING USER PROFILE LENGTH
        self.add_user_profile_length()

        # BUILDING ITEM ASSETS
        self.add_item_asset()

        # BUILDING ITEM PRICE
        self.add_item_price()

        # BUILDING ITEM SUBCLASS
        self.add_item_subclass()

        ############################

        users = list(self.xgb_dataframe.iloc[:, 0].values)
        users = np.sort(users)

        # CREATION OF GROUPS FOR XGB_RANKER
        y_train, y_test = train_test_split(list(set(users)), test_size=0.1, random_state=1)
        y_train = np.sort(y_train)
        y_test = np.sort(y_test)

        # print(y_train)
        # print(y_test)

        train_dataframe = pd.DataFrame()
        test_dataframe = pd.DataFrame()
        train_group = []
        test_group = []

        for user_id in y_train:
            to_append = self.xgb_dataframe.loc[self.xgb_dataframe['user_id'] == user_id].copy()
            train_dataframe = train_dataframe.append(to_append)
            train_group.append(20)

        for user_id in y_test:
            to_append = self.xgb_dataframe.loc[self.xgb_dataframe['user_id'] == user_id].copy()
            test_dataframe = test_dataframe.append(to_append)
            test_group.append(20)

        X_train = train_dataframe.drop(labels={'user_id', 'item_id'}, axis=1)
        X_test = test_dataframe.drop(labels={'user_id', 'item_id'}, axis=1)

        # train_dropped = self.xgb_dataframe.drop(labels={'user_id', 'item_id'}, axis=1)


        # X_train, X_test, y_train, y_test = train_test_split(train_dropped, users, test_size=0.1, random_state=1)

        # train_dropped = X_train.drop(labels={'user_id', 'item_id'}, axis=1)
        # val_dropped = X_test.drop(labels={'user_id', 'item_id'}, axis=1)


        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        dtrain.set_group(train_group)
        dtest.set_group(test_group)

        num_round = 20  # the number of training iterations (number of trees)

        xbg_ranker = xgb.XGBRanker()
        xbg_ranker.fit(dtrain, y_train, train_group)


        # xgb_regressor = xgb.XGBRegressor()
        # xgb_regressor.fit(X_train, y_train)

        # model = xgb.train(params,
        #                   dtrain,
        #                   num_round,
        #                   verbose_eval=2,
        #                   early_stopping_rounds=20)

        # print(xgb_regressor.predict(X_test))

        print(xbg_ranker.predict(dtest))
        print("user array:" + str(len(self.user_recommendations_user_id)))
        #print(" prediction array:" + str(len(model.predict())))


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
    }

    is_test = True

    runner = XGBoost(**algorithms_choice)
    runner.run(is_test)