"""
Fixed parameters of each algorithm
"""
ICFKNN = {"topK": 10, "shrink": 10}
UCFKNN = {"topK": 500, "shrink": 10}
CBFKNN = {"topK": 100, "shrink": 200}
SLIM_BPR = {"epochs": 200, "lambda_i": 0.01, "lambda_j": 0.01 }
PURE_SVD = {"num_factors": 1000, }
ALS = {"alpha_val": 25, "n_factors": 300, "regularization": 0.5, "iterations": 50}
CFW = {"iteration_limit": 50000, "damp_coeff": 0.0, "topK": 300, "add_zeros_quota": 0.0}

"""
Definitive weights kept for the submission
"""
SUBM_WEIGHTS = {
        "icfknn": 2.5,
        "ucfknn": 0.2,
        "cbfknn": 0.5,
        "slimbpr": 0.1,
        "puresvd": 2,
        "als": 1,
        "cfw": 3,
    }

"""
Weights used for the local testing
"""
IS_TEST_WEIGHTS = [{
    "icfknn": 2.5,
    "ucfknn": 0.2,
    "cbfknn": 0.5,
    "slimbpr": 0.1,
    "puresvd": 2,
    "als": 1,
    "cfw": 3,
# }, {
#     "icfknn": 3,
#     "ucfknn": 0.1,
#     "cbfknn": 0.5,
#     "slimbpr": 1.5,
#     "puresvd": 2,
#     "als": 0.8,
# }, {
#     "icfknn": 3,
#     "ucfknn": 0.2,
#     "cbfknn": 0.5,
#     "slimbpr": 1,
#     "puresvd": 3,
#     "als": 1.5,
}]
