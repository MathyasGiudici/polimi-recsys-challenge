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

def classic_tuner():
    #It's the classic parameter tuning method seen in class
    extractor = Extractor

    some_statistics(extractor)

    URM_all = extractor.get_interaction_matrix_all(extractor)

    # Cold items have no impact in the evaluation, since they have no interactions
    # Moreover, considering how item-item and user-user CF are defined, they are not relevant.
    warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
    warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

    URM_all = URM_all[:, warm_items]

    # The same holds for users
    warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
    warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

    URM_all = URM_all[warm_users, :]

    #Split training and test
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

    x_tick = [10, 50, 100, 200, 500]
    MAP_per_k = []

    for topK in x_tick:
        recommender = ItemCFKNNRecommender(URM_train)
        recommender.fit(shrink=0.0, topK=topK)

        result_dict = evaluate_algorithm(URM_test, recommender)
        MAP_per_k.append(result_dict["MAP"])

    pyplot.plot(x_tick, MAP_per_k)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()

    x_tick = [0, 10, 50, 100, 200, 500]
    MAP_per_shrinkage = []

    for shrink in x_tick:
        recommender = ItemCFKNNRecommender(URM_train)
        recommender.fit(shrink=shrink, topK=100)

        result_dict = evaluate_algorithm(URM_test, recommender)
        MAP_per_shrinkage.append(result_dict["MAP"])

    pyplot.plot(x_tick, MAP_per_shrinkage)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Shrinkage')
    pyplot.show()


def combinate_tuner():
    #It's the classic parameter tuning method seen in class
    extractor = Extractor

    some_statistics(extractor)

    URM_all = extractor.get_interaction_matrix_all(extractor)

    # Cold items have no impact in the evaluation, since they have no interactions
    # Moreover, considering how item-item and user-user CF are defined, they are not relevant.
    warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
    warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

    URM_all = URM_all[:, warm_items]

    # The same holds for users
    warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
    warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

    URM_all = URM_all[warm_users, :]

    #Split training and test
    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

    x_tick_per_k = [10, 50, 100, 200, 500, 1000]
    MAP_per_k = []

    x_tick_per_shrinkage = [0, 10, 50, 100, 200, 500, 1000]
    MAP_per_shrinkage = []

    for topK in x_tick_per_k:
        for shrink in x_tick_per_shrinkage:
            print("Trying shrink=", str(shrink), " topK=", str(topK))
            # Recommendation
            recommender = ItemCFKNNRecommender(URM_train)
            recommender.fit(shrink=shrink, topK=topK)

            #Evaluation
            result_dict = evaluate_algorithm(URM_test, recommender)
            MAP_per_k.append(result_dict["MAP"])

    pyplot.plot(x_tick_per_k, MAP_per_k)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()

    pyplot.plot(x_tick_per_shrinkage, MAP_per_shrinkage)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Shrinkage')
    pyplot.show()

if __name__ == '__main__':

    import matplotlib.pyplot as pyplot
    from Extractor import Extractor
    from Notebooks_utils.evaluation_function import evaluate_algorithm
    from ItemCFKNNRecommender import ItemCFKNNRecommender
    from Notebooks_utils.data_splitter import train_test_holdout
    import numpy as np

    #classic_tuner()
    combinate_tuner()