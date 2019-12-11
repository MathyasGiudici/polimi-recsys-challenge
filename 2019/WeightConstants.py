"""
Fixed parameters of each algorithm
"""
ICFKNN = {"topK": 30, "shrink": 56, "similarity": "asymmetric", "normalize": True, "asymmetric_alpha": 0.6599609759159238,
          "tversky_alpha": 0.8173624832240958, "tversky_beta":0.18694499695219727}
UCFKNN = {"topK": 309, "shrink": 810, "similarity": "dice", "normalize": False, "asymmetric_alpha": 0.06296343185685262,
          "tversky_alpha": 0.8912482067490604, "tversky_beta":0.1478657805022863}
CBFKNN = {"topK": 158, "shrink": 816, "similarity": "asymmetric", "normalize": True, "asymmetric_alpha": 1.791766210561791,
          "tversky_alpha": 0.17745394383013416, "tversky_beta":1.1794441251082355}

SLIM_BPR = {"topK": 433, "epochs": 1500, "symmetric": False, "sgd_mode": "adagrad", "lambda_i": 0.00014846788775948298, "lambda_j": 5.3658185438518884e-05, "learning_rate": 0.004296818192974266}

PURE_SVD = {"num_factors": 649, }

ALS = {"alpha_val": 20, "n_factors": 50, "regularization": 0.5, "iterations": 50}
CFW = {"iteration_limit": 5000, "damp_coeff": 0.0, "topK": 100, "add_zeros_quota": 0.0}
P3A = {"topK": 500, "alpha": .5}
RP3B = {"alpha":.25, "beta":0.1, "min_rating":0, "topK":500}

"""
Definitive weights kept for the submission
"""
SUBM_WEIGHTS = {'icfknn': 4.788149531158417, 'ucfknn': 0.5352507794267048, 'cbfknn': 4.9648185615891895, 'slimbpr': 2.181068924113327, 'puresvd': 4.7931831840382815, 'als': 1, 'cfw': 3, 'p3a': 2, 'rp3b': 3}

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

