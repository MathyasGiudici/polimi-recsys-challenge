from OwnUtils.Extractor import Extractor
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from KNN.ItemCFKNNRecommender import ItemCFKNNRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


class HybridAlgorithm(object):

    def __init__(self, URM_all, test_percentage, validation_percentage):
        self.URM_all = URM_all
        self.test_perc = test_percentage
        self.validation_perc = validation_percentage

    def tuning(self):

        URM_train, URM_test = train_test_holdout(self.URM_all, 1 - self.test_perc)
        URM_train, URM_validation = train_test_holdout(self.URM_all, 1 - self.validation_perc)

        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5,10])

        recommender_class = ItemCFKNNRecommender
        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation, evaluator_test)

        hyperparameters_range_dictionary = {}
        hyperparameters_range_dictionary["topK"] = [0, 10, 20, 50, 100, 150, 200, 500, 800]
        hyperparameters_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
        hyperparameters_range_dictionary["similarity"] = ["cosine"]
        hyperparameters_range_dictionary["normalize"] = [True, False]

        recommenderDictionary = SearchInputRecommenderArgs()
        SearchInputRecommenderArgs.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train]
        SearchInputRecommenderArgs.CONSTRUCTOR_KEYWORD_ARGS: {}
        SearchInputRecommenderArgs.FIT_POSITIONAL_ARGS: dict()
        SearchInputRecommenderArgs.FIT_KEYWORD_ARGS: dict()
        SearchInputRecommenderArgs.FIT_RANGE_KEYWORD_ARGS: hyperparameters_range_dictionary
        # recommenderDictionary = {SearchInputRecommenderArgs.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
        #                          SearchInputRecommenderArgs.CONSTRUCTOR_KEYWORD_ARGS: {},
        #                          SearchInputRecommenderArgs.FIT_POSITIONAL_ARGS: dict(),
        #                          SearchInputRecommenderArgs.FIT_KEYWORD_ARGS: dict(),
        #                          SearchInputRecommenderArgs.FIT_RANGE_KEYWORD_ARGS: hyperparameters_range_dictionary}

        output_root_path = "result_experiments/"
        import os

        if not os.path.exists(output_root_path):
                os.makedirs(output_root_path)

        n_cases = 2
        metric_to_optimize = 'MAP'

        best_parameters = parameterSearch.search(recommenderDictionary, n_cases,
                                                 output_root_path, metric_to_optimize)
        print(best_parameters)

    def recommender_runner(self):
        extractor = Extractor()
        URM_all = extractor.get_all_interation_matrix()
        tuner = HybridAlgorithm(URM_all, 0.9, 0.9)
        tuner.tuning()