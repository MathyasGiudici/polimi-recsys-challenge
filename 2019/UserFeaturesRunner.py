from Hybrid.WeightedHybrid import WeightedHybrid
from RecommenderByUserFeature import RecommenderByUserFeature
from OwnUtils.Extractor import Extractor
from OwnUtils.Builder import Builder
from OwnUtils.Writer import Writer
from datetime import datetime
from Utils.evaluation_function import evaluate_algorithm
import WeightConstants
import scipy.sparse as sps

import random
import Utils.Split.split_train_validation_leave_k_out as loo

report_counter = 4
submission_counter = 4

class UserFeaturesRunner(object):

    def __init__(self, users_per_region=True, users_per_age=True):
        """
        Users_per_region and users_per_age are two boolean values which specify the type of training
        that we want to perform.
        """
        self.users_per_region = users_per_region
        self.users_per_age = users_per_age

        self.is_test = None
        self.writer = Writer
        self.result_dict = None

        self.urm_train = None
        self.urm_per_region_list = None
        self.urm_per_age_list = None

        self.urm_validation = None
        self.urm_test = None
        self.urm_post_validation = None
        self.icm = None

    def run(self, is_test):
        self.is_test = is_test

        if self.is_test:
            extractor = Extractor()
            builder = Builder()
            urm = extractor.get_urm_all()
            self.icm = extractor.get_icm_all()

            # Splitting into post-validation & testing in case of parameter tuning
            matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

            self.urm_post_validation = matrices[0]
            self.urm_test = matrices[1]

            # Splitting the post-validation matrix in train & validation
            # (Problem of merging train and validation again at the end => loo twice)
            matrices_for_validation = loo.split_train_leave_k_out_user_wise(self.urm_post_validation, 1, False, True)
            self.urm_train = matrices_for_validation[0]
            self.urm_validation = matrices_for_validation[1]

            # Building the urm_per_feature lists
            if self.users_per_region:
                self.urm_per_region_list = builder.build_per_region_urm_train(self.urm_train)
            if self.users_per_age:
                self.urm_per_age_list = builder.build_per_age_urm_train(self.urm_train)

            self.write_report()
            self.evaluate()

        else:
            extractor = Extractor
            users = extractor.get_target_users_of_recs(extractor)
            self.urm_train = extractor.get_urm_all(extractor)
            self.icm = extractor.get_icm_all(extractor)

            #self.write_submission(users)


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

        self.writer.write_report(self.writer, "CBFKNN: " + str(WeightConstants.CBFKNN), report_counter)
        self.writer.write_report(self.writer, "ICFKNN: " + str(WeightConstants.ICFKNN), report_counter)
        self.writer.write_report(self.writer, "UCFKNN: " + str(WeightConstants.UCFKNN), report_counter)

        if self.users_per_region:
            self.writer.write_report(self.writer, "Used USER_PER_REGION TRAINING PROCEDURE", report_counter)
        if self.users_per_age:
            self.writer.write_report(self.writer, "Used USER_PER_AGE TRAINING PROCEDURE", report_counter)

        self.writer.write_report(self.writer, "VALIDATION", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)


    def write_submission(self, users):
        """
        This method is used to write the submission, selecting only chosen algorithms
        :return:
        """
        self.writer.write_header(self.writer, sub_counter=submission_counter)

        recommender = RecommenderByUserFeature(self.urm_post_validation, self.icm, self.urm_per_region_list,
                                               self.urm_per_age_list)
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
        generated_weights = []

        for weight in WeightConstants.IS_TEST_WEIGHTS:
            print("--------------------------------------")
            generated_weights.append(weight)

            recommender = RecommenderByUserFeature(self.urm_train, self.icm, self.urm_per_region_list,
                                                               self.urm_per_age_list, weight)
            recommender.fit()

            result_dict = evaluate_algorithm(self.urm_validation, recommender)
            results.append(float(result_dict["MAP"]))

            self.writer.write_report(self.writer, str(result_dict), report_counter)

        # Retriving correct weight
        results.sort()
        weight = generated_weights[int(results.index(max(results)))]

        self.writer.write_report(self.writer, "--------------------------------------", report_counter)
        self.writer.write_report(self.writer, "TESTING", report_counter)
        self.writer.write_report(self.writer, "--------------------------------------", report_counter)

        recommender = RecommenderByUserFeature(self.urm_post_validation, self.icm, self.urm_per_region_list,
                                               self.urm_per_age_list)
        recommender.fit()
        result_dict = evaluate_algorithm(self.urm_test, recommender)

        self.writer.write_report(self.writer, str(result_dict), report_counter)