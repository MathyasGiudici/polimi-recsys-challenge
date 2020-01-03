PURE_SVD = [
    {"num_factors": 7500}, {"num_factors": 10000}, {"num_factors": 20000}, {"num_factors": 15000},
]

ICFKNN_BEST = {'topK': 6, 'shrink': 46, 'similarity': 'tversky', 'normalize': True, 'asymmetric_alpha': 1.8835880841431558, 'tversky_alpha': 0.481142300165854, 'tversky_beta': 1.5213714818344097}
CBFKNN_BEST = {'topK': 650, 'shrink': 650}
UCFKNN_BEST = {"topK": 25, "shrink": 750}
SLIM_BPR_BEST = {'topK': 10, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}


CBFKNN = [
    {"topK": 500, "shrink": 500}, {"topK": 550, "shrink": 550}, {"topK": 600, "shrink": 600}, {"topK": 650, "shrink": 650},
    {"topK": 700, "shrink": 700}, {"topK": 750, "shrink": 750},
]

UCFKNN_SKOPT = {'topK': 738, 'shrink': 7, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': 0.1281479786771185, 'tversky_alpha': 1.4260541895140952, 'tversky_beta': 0.23240544397295199}
SLIM_BPR_SKOPT = {'topK': 5, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
ALS_SKOPT = {'alpha_val': 1.5030839679715604, 'n_factors': 997, 'regularization': 9.392629191951134, 'iterations': 44}

ALS = [
    {'alpha_val': 1.5030839679715604, 'n_factors': 10, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 25, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 50, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 75, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 100, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 150, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 250, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 500, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 750, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 1000, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 1500, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 2500, 'regularization': 10, 'iterations': 44},
    {'alpha_val': 1.5030839679715604, 'n_factors': 5000, 'regularization': 10, 'iterations': 44},
]











SLIM_BPR = [
    {'topK': 5, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 10, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 25, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 50, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 75, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 100, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 250, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 500, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 750, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 1000, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 2500, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 5000, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 7500, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 10000, 'epochs': 1000, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 5, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 10, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 25, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 50, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 75, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 100, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 250, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 500, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 750, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 1000, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 2500, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 5000, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 7500, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
    {'topK': 10000, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001},
]

# UCFKNN = [
#     {"topK": 5, "shrink": 0}, {"topK": 10, "shrink": 0}, {"topK": 25, "shrink": 0}, {"topK": 50, "shrink": 0},
#     {"topK": 75, "shrink": 0}, {"topK": 100, "shrink": 0}, {"topK": 250, "shrink": 0}, {"topK": 500, "shrink": 0},
#     {"topK": 5, "shrink": 5}, {"topK": 10, "shrink": 5}, {"topK": 25, "shrink": 5}, {"topK": 50, "shrink": 5},
#     {"topK": 75, "shrink": 5}, {"topK": 100, "shrink": 5}, {"topK": 250, "shrink": 5}, {"topK": 500, "shrink": 5},
#     {"topK": 5, "shrink": 10}, {"topK": 10, "shrink": 10}, {"topK": 25, "shrink": 10}, {"topK": 50, "shrink": 10},
#     {"topK": 75, "shrink": 10}, {"topK": 100, "shrink": 10}, {"topK": 250, "shrink": 10}, {"topK": 500, "shrink": 10},
#     {"topK": 5, "shrink": 25}, {"topK": 10, "shrink": 25}, {"topK": 25, "shrink": 25}, {"topK": 50, "shrink": 25},
#     {"topK": 75, "shrink": 25}, {"topK": 100, "shrink": 25}, {"topK": 250, "shrink": 25}, {"topK": 500, "shrink": 25},
#     {"topK": 5, "shrink": 50}, {"topK": 10, "shrink": 50}, {"topK": 25, "shrink": 50}, {"topK": 50, "shrink": 50},
#     {"topK": 75, "shrink": 50}, {"topK": 100, "shrink": 50}, {"topK": 250, "shrink": 50}, {"topK": 500, "shrink": 50},
#     {"topK": 5, "shrink": 75}, {"topK": 10, "shrink": 75}, {"topK": 25, "shrink": 75}, {"topK": 50, "shrink": 75},
#     {"topK": 75, "shrink": 75}, {"topK": 100, "shrink": 75}, {"topK": 250, "shrink": 75}, {"topK": 500, "shrink": 75},
#     {"topK": 5, "shrink": 100}, {"topK": 10, "shrink": 100}, {"topK": 25, "shrink": 100}, {"topK": 50, "shrink": 100},
#     {"topK": 75, "shrink": 100}, {"topK": 100, "shrink": 100}, {"topK": 250, "shrink": 100}, {"topK": 500, "shrink": 100},
#     {"topK": 5, "shrink": 250}, {"topK": 10, "shrink": 250}, {"topK": 25, "shrink": 250}, {"topK": 50, "shrink": 250},
#     {"topK": 75, "shrink": 250}, {"topK": 100, "shrink": 250}, {"topK": 250, "shrink": 250}, {"topK": 500, "shrink": 250},
#     {"topK": 5, "shrink": 500}, {"topK": 10, "shrink": 500}, {"topK": 25, "shrink": 500}, {"topK": 50, "shrink": 500},
#     {"topK": 75, "shrink": 500}, {"topK": 100, "shrink": 500}, {"topK": 250, "shrink": 500}, {"topK": 500, "shrink": 500},
#     {"topK": 750, "shrink": 750}, {"topK": 1000, "shrink": 750}, {"topK": 1500, "shrink": 750},
#     {"topK": 2500, "shrink": 750},
#     {"topK": 5000, "shrink": 750}, {"topK": 7500, "shrink": 750}, {"topK": 10000, "shrink": 750},
#     {"topK": 20000, "shrink": 750},
#     {"topK": 750, "shrink": 1000}, {"topK": 1000, "shrink": 1000}, {"topK": 1500, "shrink": 1000},
#     {"topK": 2500, "shrink": 1000},
#     {"topK": 5000, "shrink": 1000}, {"topK": 7500, "shrink": 1000}, {"topK": 10000, "shrink": 1000},
#     {"topK": 20000, "shrink": 1000},
#     {"topK": 750, "shrink": 1500}, {"topK": 1000, "shrink": 1500}, {"topK": 1500, "shrink": 1500},
#     {"topK": 2500, "shrink": 1500},
#     {"topK": 5000, "shrink": 1500}, {"topK": 7500, "shrink": 1500}, {"topK": 10000, "shrink": 1500},
#     {"topK": 20000, "shrink": 1500},
#     {"topK": 750, "shrink": 2500}, {"topK": 1000, "shrink": 2500}, {"topK": 1500, "shrink": 2500},
#     {"topK": 2500, "shrink": 2500},
#     {"topK": 5000, "shrink": 2500}, {"topK": 7500, "shrink": 2500}, {"topK": 10000, "shrink": 2500},
#     {"topK": 20000, "shrink": 2500},
#     {"topK": 750, "shrink": 5000}, {"topK": 1000, "shrink": 5000}, {"topK": 1500, "shrink": 5000},
#     {"topK": 2500, "shrink": 5000},
#     {"topK": 5000, "shrink": 5000}, {"topK": 7500, "shrink": 5000}, {"topK": 10000, "shrink": 5000},
#     {"topK": 20000, "shrink": 5000},
#     {"topK": 750, "shrink": 7500}, {"topK": 1000, "shrink": 7500}, {"topK": 1500, "shrink": 7500},
#     {"topK": 2500, "shrink": 7500},
#     {"topK": 5000, "shrink": 7500}, {"topK": 7500, "shrink": 7500}, {"topK": 10000, "shrink": 7500},
#     {"topK": 20000, "shrink": 7500},
#     {"topK": 750, "shrink": 10000}, {"topK": 1000, "shrink": 10000}, {"topK": 1500, "shrink": 10000},
#     {"topK": 2500, "shrink": 10000},
#     {"topK": 5000, "shrink": 10000}, {"topK": 7500, "shrink": 10000}, {"topK": 10000, "shrink": 10000},
#     {"topK": 20000, "shrink": 10000},
#     {"topK": 750, "shrink": 20000}, {"topK": 1000, "shrink": 20000}, {"topK": 1500, "shrink": 20000},
#     {"topK": 2500, "shrink": 20000},
#     {"topK": 5000, "shrink": 20000}, {"topK": 7500, "shrink": 20000}, {"topK": 10000, "shrink": 20000},
#     {"topK": 20000, "shrink": 20000},
# ]
