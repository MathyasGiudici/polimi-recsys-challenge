if __name__ == '__main__':

    # num_factors = []
    #
    # from OwnUtils.Extractor import Extractor
    # import Utils.Split.split_train_validation_leave_k_out as loo
    # ex = Extractor
    # urm_all = ex.get_interaction_matrix_all(ex)
    # icm_all = ex.get_icm_all(ex)

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

    matrices = loo.split_train_validation_leave_k_out(urm_all, 1, False, True)
    urm_train = matrices[0]
    urm_test = matrices[1]

    itemKNNCF = ItemCFKNNRecommender(urm_train)
    itemKNNCF.fit(**p_icfknn)

    itemKNNCBF = ItemCBFKNNRecommender(urm_train, icm_all)
    itemKNNCBF.fit(**p_cbfknn)

    W_sparse_CF = itemKNNCF.W_sparse
    W_sparse_CBF = itemKNNCBF.W_sparse

    W_sparse_CF_sorted = np.sort(W_sparse_CF.data.copy())
    W_sparse_CBF_sorted = np.sort(W_sparse_CBF.data.copy())

    import matplotlib.pyplot as pyplot
    pyplot.plot(W_sparse_CBF_sorted, label='CBF')
    pyplot.plot(W_sparse_CF_sorted, label='CF')
    pyplot.ylabel('Similarity cell')
    pyplot.xlabel('Value cell')
    pyplot.legend()
    pyplot.show()

