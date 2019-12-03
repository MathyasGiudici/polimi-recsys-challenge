if __name__ == '__main__':

    # num_factors = []
    #
    from OwnUtils.Extractor import Extractor
    # import Utils.Split.split_train_validation_leave_k_out as loo
    ex = Extractor
    # urm_all = ex.get_interaction_matrix_all(ex)
    icm_all = ex.get_icm_all(ex)
    #print(icm_all.shape)
    # matrices = []
    # matrices = loo.split_train_leave_k_out_user_wise(urm_all, 1, True, True)
    # urm_train = matrices[0]
    # urm_validation = matrices[1]
    # urm_test = matrices[2]
    #
    # print(urm_train)
    # print("---------------------------------")
    # print(urm_validation)
    # print("---------------------------------")
    # import scipy.sparse as sp
    # print(sp.hstack((urm_train, urm_validation)))

    # Collaborative BOOSTING FW
    import numpy as np
    from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
    from CBFKNN.ItemCBFKNNRecommender import ItemCBFKNNRecommender
    from OwnUtils.Extractor import Extractor
    import Utils.Split.split_train_validation_leave_k_out as loo

    p_icfknn = {"topK": 10, "shrink": 10}
    p_cbfknn = {"topK": 100, "shrink": 200}

    ex = Extractor
    urm_all = ex.get_urm_all(ex)
    icm_all = ex.get_icm_all(ex)

    matrices = loo.split_train_leave_k_out_user_wise(urm_all, 1, False, True)
    urm_train = matrices[0]
    urm_test = matrices[1]

    itemKNNCF = ItemCFKNNRecommender(urm_train)
    itemKNNCF.fit(**p_icfknn)

    itemKNNCBF = ItemCBFKNNRecommender(urm_train, icm_all)
    itemKNNCBF.fit(**p_cbfknn)

    W_sparse_CF = itemKNNCF.W_sparse
    W_sparse_CBF = itemKNNCBF.W_sparse

    # W_sparse_CF_sorted = np.sort(W_sparse_CF.data.copy())
    # W_sparse_CBF_sorted = np.sort(W_sparse_CBF.data.copy())
    #
    # import matplotlib.pyplot as pyplot
    # pyplot.plot(W_sparse_CF_sorted, label='CF')
    # pyplot.plot(W_sparse_CBF_sorted, label='CBF')
    # pyplot.ylabel('Similarity cell')
    # pyplot.xlabel('Value cell')
    # pyplot.legend()
    # pyplot.show()

    # Get common structure
    W_sparse_CF_structure = W_sparse_CF.copy()
    W_sparse_CF_structure.data = np.ones_like(W_sparse_CF_structure.data)

    W_sparse_CBF_structure = W_sparse_CBF.copy()
    W_sparse_CBF_structure.data = np.ones_like(W_sparse_CBF_structure.data)

    W_sparse_common = W_sparse_CF_structure.multiply(W_sparse_CBF_structure)

    # Get common value between two matrices
    W_sparse_delta = W_sparse_CBF.copy().multiply(W_sparse_common)
    W_sparse_delta -= W_sparse_CF.copy().multiply(W_sparse_common)

    # W_sparse_delta_sorted = np.sort(W_sparse_delta.data.copy())
    #
    # pyplot.plot(W_sparse_delta_sorted, label="delta")
    # pyplot.ylabel('Similarity cell')
    # pyplot.xlabel('Value cell')
    # pyplot.legend()
    # pyplot.show()

    W_sparse_delta = W_sparse_delta.tocoo()

    # item_index_1 = W_sparse_delta.row[666]
    # item_index_2 = W_sparse_delta.col[666]
    #
    # print("Indices are {} and {}".format(item_index_1, item_index_2))
    #
    # print("CF similarity value is {}".format(W_sparse_CF[item_index_1, item_index_2]))
    # print("CBF similarity value is {}".format(W_sparse_CBF[item_index_1, item_index_2]))

    from Utils.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
    from Utils.Base.Evaluation.Evaluator import EvaluatorHoldout
    import Utils.evaluation_function as ev

    CFW_weighting = CFW_D_Similarity_Linalg(urm_train, icm_all, W_sparse_CF)
    CFW_weighting.fit()

    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
    results_dict, _ = evaluator_test.evaluateRecommender(itemKNNCF)
    print(results_dict)

    results_dict, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
    print(results_dict)

    results_dict, _ = evaluator_test.evaluateRecommender(CFW_weighting)
    print(results_dict)


