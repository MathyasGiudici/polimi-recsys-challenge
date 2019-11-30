import matplotlib.pyplot as pyplot
import numpy as np

from Utils.evaluation_function import evaluate_algorithm
from Utils.Split.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import Utils.Split.split_train_validation_leave_k_out as loo
from Utils.Split.DataReader_utils import remove_empty_rows_and_cols

from OwnUtils.Extractor import Extractor


class Tuner(object):

    def __init__(self, splitType="loo", needValidation=True):
        extractor = Extractor

        self.urm_all = extractor.get_urm_all(extractor)
        self.urm_all = remove_empty_rows_and_cols(self.urm_all)[0]

        self.urm_train = None
        self.urm_test = None
        self.urm_validation = None

        self.some_statistics(self, extractor)

        self.splitType = splitType
        self.needValidation = needValidation
        self.splitter(self)
        self.parameters_creation(self)

    def list_id_stats(self, ID_list, label):
        min_val = min(ID_list)
        max_val = max(ID_list)
        unique_val = len(set(ID_list))
        missing_val = 1 - unique_val / (max_val - min_val)

        print("{} data, ID: min {}, max {}, unique {}, missig {:.2f} %".format(label, min_val, max_val, unique_val,
                                                                               missing_val * 100))

    def some_statistics(self,extractor):
        userList = list(extractor.get_interaction_users(extractor))
        itemList = list(extractor.get_interaction_items(extractor))

        self.list_id_stats(self, userList, "User")
        self.list_id_stats(self, itemList, "Item")

    def splitter(self):
        if self.splitType == "percentage":
            if self.needValidation:
                self.urm_train, self.urm_test = split_train_in_two_percentage_global_sample(self.urm_all, train_percentage=0.1)
                self.urm_train, self.urm_validation = split_train_in_two_percentage_global_sample(self.urm_train, train_percentage=0.1)
            else:
                self.urm_train, self.urm_test = split_train_in_two_percentage_global_sample(self.urm_all, train_percentage=0.2)
        elif self.splitType == "loo":
            matrices = loo.split_train_leave_k_out_user_wise(self.urm_all, 1, self.needValidation, True)

            self.urm_train = matrices[0]
            self.urm_test = matrices[1]

            if self.needValidation:
                self.urm_validation = matrices[2]

        else:
            print("splitType in function spiltter() not defined")

    def parameters_creation(self):
        # Defining all variables
        self.tick_per_k = [10, 50, 100, 200, 500, 1000]
        self.tick_per_shrinkage = [0, 10, 50, 100, 200, 500]
        self.MAP_per_k = []
        self.MAP_per_shrinkage = []


    def classic_tuner(self):

        self.parameters_creation(self)

        from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
        recommender = ItemCFKNNRecommender(self.urm_train)

        best_k = 0
        current_map = -1

        for topK in self.tick_per_k:
            print("---------------- topK: ", str(topK), " shrink: ", str(0), " ----------------")
            recommender.fit(shrink=0.0, topK=topK)

            if self.needValidation:
                result_dict = evaluate_algorithm(self.urm_validation, recommender)
            else:
                result_dict = evaluate_algorithm(self.urm_test,recommender)

            if result_dict["MAP"] > current_map:
                current_map = result_dict["MAP"]
                best_k = topK

            self.MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(self.tick_per_k, self.MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        best_s = 0
        current_map = -1

        for shrink in self.tick_per_shrinkage:
            print("---------------- topK: ", str(100), " shrink: ", str(shrink), " ----------------")
            recommender.fit(shrink=shrink, topK=100)

            if self.needValidation:
                result_dict = evaluate_algorithm(self.urm_validation, recommender)
            else:
                result_dict = evaluate_algorithm(self.urm_test, recommender)

            if result_dict["MAP"] > current_map:
                current_map = result_dict["MAP"]
                best_s = shrink

            self.MAP_per_shrinkage.append(result_dict["MAP"])

        pyplot.plot(self.tick_per_shrinkage, self.MAP_per_shrinkage)
        pyplot.ylabel('MAP')
        pyplot.xlabel('Shrinkage')
        pyplot.show()

        self.one_shot_test(self, best_k, best_s)

    def parallel_tuner(self):

        self.parameters_creation(self)

        from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
        recommender = ItemCFKNNRecommender(self.urm_train)

        best_k = 0
        best_s = 0
        current_map = -1

        for topK in self.tick_per_k:
            for shrink in self.tick_per_shrinkage:
                print("---------------- topK: ", str(topK), " shrink: ", str(shrink), " ----------------")
                # Recommendation
                recommender.fit(shrink=shrink, topK=topK)

                # Evaluation
                if self.needValidation:
                    result_dict = evaluate_algorithm(self.urm_validation, recommender)
                else:
                    result_dict = evaluate_algorithm(self.urm_test, recommender)

                self.MAP_per_k.append(result_dict["MAP"])

                if result_dict["MAP"] > current_map:
                    current_map = result_dict["MAP"]
                    best_k = topK
                    best_s = shrink

            pyplot.plot(self.tick_per_shrinkage, self.MAP_per_k)
            pyplot.ylabel('MAP')
            string = 'Shrinkage, with TopK:' + str(topK)
            pyplot.xlabel(string)
            pyplot.show()

            self.MAP_per_k.clear()

        self.one_shot_test(self,best_k, best_s)

    def one_shot_test(self, topK: float, shrink: float) -> None:

        from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
        recommender = ItemCFKNNRecommender(self.urm_train)

        recommender.fit(shrink=shrink, topK=topK)

        print("---------------- topK: ", str(topK), " shrink: ", str(shrink), " ----------------")

        if self.needValidation:
            evaluate_algorithm(self.urm_validation, recommender)
        else:
            evaluate_algorithm(self.urm_test, recommender)

