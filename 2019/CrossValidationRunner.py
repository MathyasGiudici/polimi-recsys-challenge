from Hybrid.WeightedHybrid import WeightedHybrid
from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm, evaluate_algorithm_crossvalidation
import ParametersTuning
import WeightConstants

import random
import Utils.Split.split_train_validation_leave_k_out as loo

"""
Specify the report and the submission in which we will write the results
"""
report_counter = 10
submission_counter = 3

class CrossValidationRunner(object):

    def __init__(self, cbfknn=True, icfknn=True, ucfknn=True, slim_bpr=True, pure_svd=True, als=True, cfw=True,
                 p3a=True, rp3b=True):
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
        self.p3a = p3a
        self.rp3b = rp3b

        self.is_test = None
        self.writer = Writer
        self.extractor = Extractor()
        self.result_dict = None

        self.urm_train = None
        self.urm_validation = None
        self.icm = None

        self.p_cbfknn = None
        self.p_icfknn = None
        self.p_ucfknn = None
        self.p_slimbpr = None
        self.p_puresvd = None
        self.p_als = None
        self.p_cfw = None
        self.p_p3a = None
        self.p_rp3b = None

        self.target_users = []
        self.results = []

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
        if self.p3a:
            self.p_p3a = WeightConstants.P3A
        if self.rp3b:
            self.p_rp3b = WeightConstants.RP3B

        self.MAPs = []


    def run(self, is_test):
        """
        From here we start each algorithm.
        :param is_test: specifies if we want to write a report or a submission
        """
        self.is_test = is_test
        self.icm = self.extractor.get_icm_all()

        if self.is_test:

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

            # TUNING WITH THE DIFFERENT PARAMS
            for params in ParametersTuning.UCFKNN:
                if self.ucfknn:
                    self.p_ucfknn = params
                self.write_report()

                # URM splitted in 4 smaller URMs for cross-validation
                for i in range(0, 4):
                    self.urm_validation = vals[i].copy()
                    self.urm_train = urms[i].copy()
                    self.target_users = self.extractor.get_target_users_of_specific_part(i+1)

                    self.evaluate(i+1, target_profiles[i])

                self.output_average_MAP()

            self.output_best_params()

        else:
            self.p_cbfknn = ParametersTuning.CBFKNN_BEST
            self.p_icfknn = ParametersTuning.ICFKNN_BEST

            users = self.extractor.get_target_users_of_recs()
            self.urm_train = self.extractor.get_urm_all()

            self.write_submission(users)


    def write_report(self):
        """
        This method is useful to write the report, selecting only chosen algorithms
        """
        now = datetime.now()
        date = datetime.fromtimestamp(datetime.timestamp(now))

        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------\n", report_counter)
        self.writer.write_report(self.writer, "REPORT " + str(date), report_counter)
        self.writer.write_report(self.writer, "----- Cross-validation procedure -----\n", report_counter)
        self.writer.write_report(self.writer, "Fixed parameters", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)

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
        if self.p3a:
            self.writer.write_report(self.writer, "P3A: " + str(self.p_p3a), report_counter)
        if self.rp3b:
            self.writer.write_report(self.writer, "P3A: " + str(self.p_rp3b), report_counter)
        self.writer.write_report(self.writer, "--------------------------------------\n", report_counter)


    def write_submission(self, users):
        """
        This method is used to write the submission, selecting only chosen algorithms
        :return:
        """
        self.writer.write_header(self.writer, sub_counter=submission_counter)

        recommender = WeightedHybrid(self.urm_train, self.icm, self.p_icfknn, self.p_ucfknn, self.p_cbfknn,
                                     self.p_slimbpr, self.p_puresvd, self.p_als, self.p_cfw, self.p_p3a, self.p_rp3b,
                                     WeightConstants.NO_WEIGHTS[0], seen_items=self.urm_train)
        recommender.fit()

        from tqdm import tqdm

        for user_id in tqdm(users):
            recs = recommender.recommend(user_id, at=10)
            self.writer.write(self.writer, user_id, recs, sub_counter=submission_counter)

        print("Submission file written")


    def evaluate(self, index: int, target_users_profile):
        """
        Method used for the validation and the calculation of the weights
        """
        generated_weights = []

        self.writer.write_report(self.writer, "VALIDATION " + str(index), report_counter)

        for weight in self.get_test_weights(add_random=False):

            generated_weights.append(weight)
            print("--------------------------------------")

            recommender = WeightedHybrid(self.urm_train, self.icm.copy(), self.p_icfknn, self.p_ucfknn, self.p_cbfknn,
                                         self.p_slimbpr, self.p_puresvd, self.p_als, self.p_cfw, self.p_p3a,
                                         self.p_rp3b, weight, seen_items=target_users_profile)
            recommender.fit()
            result_dict = evaluate_algorithm_crossvalidation(self.urm_validation, recommender, self.target_users)
            self.results.append(float(result_dict["MAP"]))

            # self.writer.write_report(self.writer, str(weight), report_counter)
            self.writer.write_report(self.writer, str(result_dict), report_counter)
            self.writer.write_report(self.writer, "--------------------------------------", report_counter)

        # Retriving correct weight
        # results.sort()
        # weight = generated_weights[int(self.results.index(max(self.results)))]


    def get_test_weights(self, add_random=False):
        if not add_random:
            return WeightConstants.NO_WEIGHTS
        else:
            new_weights = []
            for weight in WeightConstants.IS_TEST_WEIGHTS:
                new_weights.append(weight)
                for i in range(0, 5):
                    new_obj = weight.copy()
                    new_obj["icfknn"] += round(random.uniform(- min(0.5, weight["icfknn"]), 0.5), 2)
                    new_obj["ucfknn"] += round(random.uniform(- min(0.5, weight["ucfknn"]), 0.5), 2)
                    new_obj["cbfknn"] += round(random.uniform(- min(0.5, weight["cbfknn"]), 0.5), 2)
                    new_obj["slimbpr"] += round(random.uniform(- min(0.5, weight["slimbpr"]), 0.5), 2)
                    new_obj["puresvd"] += round(random.uniform(- min(0.5, weight["puresvd"]), 0.5), 2)
                    new_obj["als"] += round(random.uniform(- min(0.5, weight["als"]), 0.5), 2)
                    new_obj["cfw"] += round(random.uniform(- min(0.5, weight["cfw"]), 0.5), 2)
                    new_obj["p3a"] += round(random.uniform(- min(0.5, weight["p3a"]), 0.5), 2)
                    new_obj["rp3b"] += round(random.uniform(- min(0.5, weight["rp3b"]), 0.5), 2)
                    new_weights.append(new_obj)

            return new_weights


    def output_average_MAP(self):
        average_MAP = 0
        for res in self.results:
            average_MAP += res

        average_MAP /= len(self.results)

        self.MAPs.append(average_MAP)

        self.results.clear()

        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "The average MAP is: " + str(average_MAP), report_counter)


    def output_best_params(self):
        best_MAP = max(self.MAPs)
        index = self.MAPs.index(best_MAP)
        best_params = ParametersTuning.UCFKNN[index]
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "With a MAP of " + str(best_MAP) + " the best parameters are: " +
                                 str(best_params), report_counter)



