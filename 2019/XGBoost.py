from Hybrid.WeightedHybrid import WeightedHybrid
from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm
import WeightConstants
from Utils.Base.NonPersonalizedRecommender import TopPop

import random
import Utils.Split.split_train_validation_leave_k_out as loo

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
        cutoff = 20
        user_recommendations_user_id = []
        user_recommendations_items = []

        for n_user in range(0, self.urm_test.shape[0]):
            recommendations = recommender.recommend(n_user, at=cutoff)

            user_recommendations_user_id.extend([n_user] * len(recommendations))
            user_recommendations_items.extend(recommendations)

        # BUILDING DATAFRAME
        import pandas as pd
        import numpy as np

        train_dataframe = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

        # BUILDING POPULARITY ITEMS
        topPop = TopPop(self.urm_train)
        topPop.fit()

        topPop_score_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            topPop_score = topPop._compute_item_score([user_id])[0, item_id]
            topPop_score_list.append(topPop_score)

        train_dataframe['item_popularity'] = pd.Series(topPop_score_list, index=train_dataframe.index)

        # BUILDING USER PROFILE LENGTH
        user_profile_len = np.ediff1d(self.urm_train.indptr)

        user_profile_len_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
            user_profile_len_list.append(user_profile_len[user_id])

        train_dataframe['user_profile_len'] = pd.Series(user_profile_len_list, index=train_dataframe.index)

        feature_1_list = []

        for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):

            item_features = self.icm[item_id, :]

            #if target_feature in item_features.indices:

            if len(item_features.indices) != 0:
                feature_1_list.append(self.icm[item_id,item_features.indices[0]])
            else:
                feature_1_list.append(0)

        train_dataframe['item_feature_1'] = pd.Series(feature_1_list, index=train_dataframe.index)

        print(train_dataframe[0:10])
        return

        import xgboost as xgb

        params = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # step for each iteration
            'silent': 1,  # keep it quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 3,  # the number of classes
            'eval_metric': 'merror'}  # evaluation metric

        num_round = 20  # the number of training iterations (number of trees)

        model = xgb.train(params,
                          train_dataframe,
                          num_round)

        print(model.predict())
        print("user array:" + str(len(user_recommendations_user_id)))
        print(" prediction array:" + str(len(model.predict())))

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