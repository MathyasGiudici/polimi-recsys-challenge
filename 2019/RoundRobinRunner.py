from Hybrid.RoundRobinHybrid import RoundRobinHybrid
from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm
import WeightConstants

import random
import Utils.Split.split_train_validation_leave_k_out as loo

"""
Specify the report and the submission in which we will write the results
"""
report_counter = 0
submission_counter = 0

variated_version = False

class RoundRobinRunner(object):

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
            extractor = Extractor
            urm = extractor.get_urm_all(extractor)
            self.icm = extractor.get_icm_all(extractor)

            # Splitting into post-validation & testing in case of parameter tuning
            matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

            self.urm_post_validation = matrices[0]
            self.urm_test = matrices[1]

            # Splitting the post-validation matrix in train & validation
            # (Problem of merging train and validation again at the end => loo twice)
            matrices_for_validation = loo.split_train_leave_k_out_user_wise(self.urm_post_validation, 1, False, True)
            self.urm_train = matrices_for_validation[0]
            self.urm_validation = matrices_for_validation[1]

            self.write_report()
            self.evaluate()

        else:
            extractor = Extractor
            users = extractor.get_target_users_of_recs(extractor)
            self.urm_train = extractor.get_urm_all(extractor)
            self.icm = extractor.get_icm_all(extractor)

            self.write_submission(users)


    def write_report(self):
        """
        This method is useful to write the report, selecting only chosen algorithms
        """
        now = datetime.now()
        date = datetime.fromtimestamp(datetime.timestamp(now))

        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------\n", report_counter)
        self.writer.write_report(self.writer, "REPORT " + str(date) + "\n", report_counter)
        self.writer.write_report(self.writer, "Fixed parameters", report_counter)

        if self.cbfknn:
            self.writer.write_report(self.writer, "CBFKNN: " + str(self.p_cbfknn), report_counter)
        if self.icfknn:
            self.writer.write_report(self.writer, "ICFKNN: " + str(self.p_icfknn), report_counter)
        if self.ucfknn:
            self.writer.write_report(self.writer, "UCFKNN: " + str(self.p_ucfknn), report_counter)
        if self.slim_bpr:
            self.writer.write_report(self.writer, "SLIM_BPR: " + str(self.p_slimbpr), report_counter)
        if self.pure_svd:
            self.writer.write_report(self.writer, "PURE_SVD: " + str(self.p_puresvd), report_counter)
        if self.als:
            self.writer.write_report(self.writer, "ALS: " + str(self.p_als), report_counter)
        if self.cfw:
            self.writer.write_report(self.writer, "CFW: " + str(self.p_cfw), report_counter)

        self.writer.write_report(self.writer, "\n", report_counter)
        self.writer.write_report(self.writer, "VALIDATION", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)


    def write_submission(self, users):
        """
        This method is used to write the submission, selecting only chosen algorithms
        :return:
        """
        self.writer.write_header(self.writer, sub_counter=submission_counter)

        recommender = RoundRobinHybrid(self.urm_train, self.icm, self.p_icfknn, self.p_ucfknn, self.p_cbfknn, self.p_slimbpr,
                             self.p_puresvd, self.p_als, self.p_cfw, WeightConstants.SUBM_WEIGHTS, variated_version)
        recommender.fit()

        from tqdm import tqdm

        for user_id in tqdm(users):
            recs = recommender.recommend(user_id, at=10)
            self.writer.write(self.writer, user_id, recs, sub_counter=submission_counter)

        print("Submission file written")


    def evaluate(self):
        """
        Method used for the validation and the calculation of the weights
        """
        results = []
        orders = []

        for order in WeightConstants.SUB_TEST_ROUNDROBIN:
            print("--------------------------------------")
            orders.append(order.copy())

            recommender = RoundRobinHybrid(self.urm_train, self.icm, self.p_icfknn, self.p_ucfknn, self.p_cbfknn,
                                     self.p_slimbpr, self.p_puresvd, self.p_als, self.p_cfw, order, variated_version)
            recommender.fit()
            result_dict = evaluate_algorithm(self.urm_validation, recommender)
            results.append(float(result_dict["MAP"]))

            self.writer.write_report(self.writer, str(order), report_counter)
            self.writer.write_report(self.writer, str(result_dict), report_counter)

        # Retriving correct weight
        results.sort()
        order = orders[int(results.index(max(results)))]

        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "TESTING", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)

        recommender = RoundRobinHybrid(self.urm_post_validation, self.icm, self.p_icfknn, self.p_ucfknn, self.p_cbfknn,
                             self.p_slimbpr, self.p_puresvd, self.p_als, self.p_cfw, order, variated_version)
        recommender.fit()
        result_dict = evaluate_algorithm(self.urm_test, recommender)

        self.writer.write_report(self.writer, str(order), report_counter)
        self.writer.write_report(self.writer, str(result_dict), report_counter)



