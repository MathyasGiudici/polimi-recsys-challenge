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

import xgboost as xgb
import pandas as pd
import numpy as np

"""
Specify the report and the submission in which we will write the results
"""
report_counter = 10
submission_counter = 2


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
        self.train_dataframe = None

        self.builder = Builder()

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
        self.train_dataframe = pd.DataFrame({"user_id": self.user_recommendations_user_id, "item_id": self.user_recommendations_items})

        ############################
        ###### ADDING FEATURES #####
        ############################

        # BUILDING POPULARITY ITEMS
        #self.add_top_pop_items()

        # BUILDING USER PROFILE LENGTH
        #self.add_user_profile_length()

        # BUILDING ITEM ASSETS
        self.add_item_asset()

        # BUILDING ITEM PRICE
        self.add_item_price()

        # BUILDING ITEM SUBCLASS
        self.add_item_subclass()


        print(self.train_dataframe.head())
        #return

        params = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # step for each iteration
            'silent': 1,  # keep it quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            #'num_class': 3,  # the number of classes
            'eval_metric': 'merror'}  # evaluation metric

        num_round = 20  # the number of training iterations (number of trees)

        msk = np.random.rand(len(self.train_dataframe)) < 0.8
        dtrain = self.train_dataframe[msk]
        dtest = self.train_dataframe[~msk]

        dtrain = xgb.DMatrix(dtrain, missing=-999.0)
        dtest = xgb.DMatrix(dtest, missing=-999.0)

        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        model = xgb.train(params, dtrain, num_round, evallist)

        print(model.predict())
        #print("user array:" + str(len()))
        print(" prediction array:" + str(len(model.predict())))


    def add_top_pop_items(self):
        # BUILDING POPULARITY ITEMS
        print("Adding TopPop items feature...")

        topPop = TopPop(self.urm_train)
        topPop.fit()

        topPop_score_list = []

        for user_id, item_id in zip(self.user_recommendations_user_id, self.user_recommendations_items):
            topPop_score = topPop._compute_item_score([user_id])[0, item_id]
            topPop_score_list.append(topPop_score)

        self.train_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=self.train_dataframe.index)
        print("Addition completed!")

    def add_user_profile_length(self):
        # BUILDING USER PROFILE LENGTH
        print("Adding user profile length feature...")

        # user_profile_len = np.ediff1d(self.urm_train.indptr)

        user_profile_len_list = []

        from tqdm import tqdm
        for user_id, item_id in zip(self.user_recommendations_user_id, self.user_recommendations_items):
            user_profile_len_list.append(len(self.urm_train[user_id].indices))

        self.train_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=self.train_dataframe.index)
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
                assets.append(-1)

        icm_asset_list = []

        for item_id in self.user_recommendations_items:
            icm_asset_list.append(assets[item_id])

        self.train_dataframe['item_asset'] = pd.Series(icm_asset_list, index=self.train_dataframe.index)
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

        self.train_dataframe['item_price'] = pd.Series(icm_asset_list, index=self.train_dataframe.index)
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

        self.train_dataframe['item_subclass'] = pd.Series(icm_asset_list, index=self.train_dataframe.index)
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
    #runner.add_item_subclass()
    runner.run(is_test)