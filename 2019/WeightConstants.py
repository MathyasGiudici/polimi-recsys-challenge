"""
Fixed parameters of each algorithm
"""
ICFKNN = {'topK': 800, 'shrink': 130, 'similarity': 'dice', 'normalize': False, 'asymmetric_alpha': 2.0, 'tversky_alpha': 0.06645449914150482, 'tversky_beta': 0.0}
UCFKNN = {'topK': 124, 'shrink': 1000, 'similarity': 'jaccard', 'normalize': False, 'asymmetric_alpha': 2.0, 'tversky_alpha': 1.4568982966977921, 'tversky_beta': 1.1915570273797769}
CBFKNN = {'topK': 800, 'shrink': 1000, 'similarity': 'tversky', 'normalize': True, 'asymmetric_alpha': 0.8403440634469317, 'tversky_alpha': 2.0, 'tversky_beta': 0.0}

SLIM_BPR = {'topK': 5, 'epochs': 20, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0016383008979062898}

PURE_SVD = {'num_factors': 481}

ALS = {"alpha_val": 20, "n_factors": 50, "regularization": 0.5, "iterations": 50}
CFW = {"iteration_limit": 5000, "damp_coeff": 0.0, "topK": 100, "add_zeros_quota": 0.0}

P3A = {'topK': 800, 'alpha': 0.0, 'normalize_similarity': True}
RP3B = {'topK': 5, 'alpha': 0.0, 'beta': 0.0, 'normalize_similarity': True}

"""
Definitive weights kept for the submission
"""
SUBM_WEIGHTS = {'icfknn': 2.5802194484369156, 'ucfknn': 0.0, 'cbfknn': 5.0, 'slimbpr': 2.7268466077091325, 'puresvd': 0.6702145369478825, 'p3a': 1.9862766182019822, 'rp3b': 5.0}

_SUBM_WEIGHTS = {
        "icfknn": 2.5,
        "ucfknn": 0.2,
        "cbfknn": 1,
        "slimbpr": 1.5,
        "puresvd": 2,
        "als": 1,
        "cfw": 3,
    }

"""
Weights used for the local testing
"""
IS_TEST_WEIGHTS = [ {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 1, 'rp3b': 1} ]
_IS_TEST_WEIGHTS = [
    {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 2, 'rp3b': 3},
    {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 2, 'rp3b': 3.5},
    {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 2.5, 'rp3b': 3.5},
    {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 1.5, 'rp3b': 2.5},
    {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 1.5, 'rp3b': 3},
]



SUBM_ROUNDROBIN = [5, 4, 3, 2, 1, 0]

SUB_TEST_ROUNDROBIN = [#[0, 1, 2, 3, 4, 5],
                       #[0, 4, 3, 5, 2, 1],
                       #[4, 0, 3, 5, 2, 1],
                       #[0, 3, 4, 5, 2, 1],
                       [4, 3, 0, 5, 2, 1],
                       [5, 4, 3, 2, 1, 0],
                                #]

#SUB_TEST_ROUNDROBIN = [
                        [0, 3, 4, 5, 2, 1],
                        # [0, 1, 2, 3, 4, 5],
                        # [0, 4, 3, 5, 2, 1],
                        [0, 3, 1, 4, 5, 2],
                        [0, 1, 3, 4, 5, 2],
                        [0, 3, 4, 1, 5, 2],
                        [0, 3, 1, 4, 2, 5],
                        [0, 1, 3, 4, 2, 5],
                        [0, 3, 4, 1, 2, 5]]

