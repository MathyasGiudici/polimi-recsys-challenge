import matplotlib.pyplot as pyplot
from Extractor import Extractor
from Notebooks_utils.evaluation_function import evaluate_algorithm
from ItemCFKNNRecommender import ItemCFKNNRecommender
from Notebooks_utils.data_splitter import train_test_holdout
import Data_manager.Split_functions.split_train_validation_leave_k_out as loo
from Data_manager.DataReader_utils import remove_empty_rows_and_cols
import numpy as np
from Splitter import Splitter


def some_statistics(extractor):
    userList = list(extractor.get_interaction_users(extractor))
    itemList = list(extractor.get_interaction_items(extractor))

    list_ID_stats(userList, "User")
    list_ID_stats(itemList, "Item")


def list_ID_stats(ID_list, label):
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_val = len(set(ID_list))
    missing_val = 1 - unique_val / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missig {:.2f} %".format(label, min_val, max_val, unique_val,
                                                                           missing_val * 100))


def classic_tuner(typeSplit='percentage_split'):
    # It's the classic parameter tuning method seen in class
    extractor = Extractor

    some_statistics(extractor)

    URM_all = extractor.get_interaction_matrix_all(extractor)

    URM_train = None
    URM_test = None
    URM_validation = None

    if typeSplit == 'percentage_split':
        # Cold items have no impact in the evaluation, since they have no interactions
        # Moreover, considering how item-item and user-user CF are defined, they are not relevant.
        warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
        warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

        URM_all = URM_all[:, warm_items]

        # The same holds for users
        warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
        warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

        URM_all = URM_all[warm_users, :]

        # Split training and test
        URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

    elif typeSplit == 'leave_one_out_split':

        validation_set_needed = True
        matrices = loo.split_train_leave_k_out_user_wise(URM_all.todense(), 1,
                                                         validation_set_needed, True)

        if validation_set_needed:
            URM_train = matrices[0]
            URM_test = matrices[1]
            URM_validation = matrices[2]
        else:
            URM_train = matrices[0]
            URM_test = matrices[1]
            URM_validation = URM_test

    x_tick_per_k = [10, 50, 100, 200, 500, 1000]
    MAP_per_k = []

    for topK in x_tick_per_k:
        recommender = ItemCFKNNRecommender(URM_train)
        recommender.fit(shrink=0.0, topK=topK)

        print("topK: ", str(topK), " shrink: ", str(0.0))

        result_dict = evaluate_algorithm(URM_validation, recommender)
        MAP_per_k.append(result_dict["MAP"])

    pyplot.plot(x_tick_per_k, MAP_per_k)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()

    x_tick_per_shrinkage = [0, 10, 50, 100, 200, 500]
    MAP_per_shrinkage = []

    for shrink in x_tick_per_shrinkage:
        recommender = ItemCFKNNRecommender(URM_train)
        recommender.fit(shrink=shrink, topK=100)

        print("topK: ", str(100), " shrink: ", str(shrink))

        result_dict = evaluate_algorithm(URM_validation, recommender)
        MAP_per_shrinkage.append(result_dict["MAP"])

    pyplot.plot(x_tick_per_shrinkage, MAP_per_shrinkage)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Shrinkage')
    pyplot.show()


def combinate_tuner(similarity):
    # It's the classic parameter tuning method seen in class
    extractor = Extractor

    some_statistics(extractor)

    URM_all = extractor.get_interaction_matrix_all(extractor)

    URM_all = remove_empty_rows_and_cols(URM_all)[0]

    matrices = loo.split_train_leave_k_out_user_wise(URM_all.todense(), 1, True, True)

    URM_train = matrices[0]
    URM_test = matrices[1]
    URM_validation = matrices[2]

    x_tick_per_k = [10, 50, 100, 200, 500, 1000, 2000]
    x_tick_per_shrinkage = [0, 10, 50, 100, 200, 500, 1000]

    MAP = []

    best_k = 0
    best_s = 0

    currentMAP = -1

    for topK in x_tick_per_k:
        for shrink in x_tick_per_shrinkage:
            print("Trying topK=", str(topK), " shrink=", str(shrink))
            # Recommendation
            recommender = ItemCFKNNRecommender(URM_train)
            recommender.fit(shrink=shrink, topK=topK, similarity=similarity)

            # Evaluation
            result_dict = evaluate_algorithm(URM_validation, recommender)
            MAP.append(result_dict["MAP"])

            if result_dict["MAP"] > currentMAP:
                currentMAP = result_dict["MAP"]
                best_k = topK
                best_s = shrink

        pyplot.plot(x_tick_per_shrinkage, MAP)
        pyplot.ylabel('MAP')
        string = 'Shrinkage, with TopK:' + str(topK)
        pyplot.xlabel(string)
        pyplot.show()

        MAP.clear()

    test_after_tuning(best_k, best_s, URM_train, URM_test, similarity)

def test_after_tuning(topK, shrink, URM_train, URM_test,similarity):

    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=topK, similarity = similarity)

    print("AFTER TUNING: topK: ", str(topK), " shrink: ", str(shrink))

    result_dict = evaluate_algorithm(URM_test, recommender)


def test_base():
    # It's the classic parameter tuning method seen in class
    extractor = Extractor

    some_statistics(extractor)

    URM_all = extractor.get_interaction_matrix_all(extractor)
    URM_all = remove_empty_rows_and_cols(URM_all)[0]
    matrices = loo.split_train_leave_k_out_user_wise(URM_all.todense(), 1, False, True)

    URM_train = matrices[0]
    URM_test = matrices[1]

    recommender = ItemCFKNNRecommender(URM_train)
    recommender.fit(shrink=10, topK=1000)

    result_dict = evaluate_algorithm(URM_test, recommender)


if __name__ == '__main__':
    # test_after_tuning(10, 10)
    #classic_tuner('leave_one_out_split')
    combinate_tuner("dice")
    combinate_tuner("jaccard")
    #test_base()