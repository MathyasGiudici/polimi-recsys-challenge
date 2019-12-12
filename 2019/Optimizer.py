import WeightConstants
from OwnUtils.Extractor import Extractor
import Utils.Split.split_train_validation_leave_k_out as loo
from Utils.evaluation_function import evaluate_algorithm
from Hybrid.WeightedHybrid import WeightedHybrid
from OwnUtils.Writer import Writer

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

similarity_type = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


class Optimizer(object):

    def __init__(self):
        self.HYP = {}
        self.report_counter = 50
        self.writer = Writer()

        # Some parameters
        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()
        self.hyperparams_single_value = dict()

        # Extractor for matricies
        extractor = Extractor()
        urm = extractor.get_urm_all()
        self.icm = extractor.get_icm_all()

        # Splitting into post-validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)
        self.urm_train = matrices[0]
        self.urm_test = matrices[1]

    def optimeze_weights(self):
        # weights = {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 2, 'rp3b': 3}
        weights = {}
        weights["icfknn"] = Real(low=0, high=5, prior='uniform') # high=100000, prior='log-uniform')
        weights["ucfknn"] = Real(low=0, high=5, prior='uniform')
        weights["cbfknn"] = Real(low=0, high=5, prior='uniform')
        weights["slimbpr"] = Real(low=0, high=5, prior='uniform')
        weights["puresvd"] = Real(low=0, high=5, prior='uniform')
        #weights["als"] = Real(low=0, high=5, prior='uniform')
        weights["p3a"] = Real(low=0, high=5, prior='uniform')
        weights["rp3b"] = Real(low=0, high=5, prior='uniform')

        return weights

    def rebuild_weights(self, array):
        return {"icfknn": array[0], "ucfknn": array[1], "cbfknn": array[2], "slimbpr": array[3],
                "puresvd": array[4], "p3a": array[5], "rp3b": array[6]}

    def optimize_single_KNN(self):
        parameters = {"topK": Integer(5, 800), "shrink": Integer(0, 1000), "similarity": Categorical(similarity_type),
                      "normalize": Categorical([True, False])}

        if parameters["similarity"] == "asymmetric":
            parameters["normalize"] = Categorical([True])

        elif parameters["similarity"] == "tversky":
            parameters["normalize"] = Categorical([True])

        parameters["asymmetric_alpha"] = Real(low=0, high=2, prior='uniform')
        parameters["tversky_alpha"] = Real(low=0, high=2, prior='uniform')
        parameters["tversky_beta"] = Real(low=0, high=2, prior='uniform')

        return parameters

    def rebuild_single_KNN(self, array):
        return {"topK": array[0], "shrink": array[1], "similarity": array[2], "normalize": array[3],
                      "asymmetric_alpha": array[4], "tversky_alpha": array[5], "tversky_beta": array[6]}

    def optimize_all_KNN(self):
        ICFKNN = self.optimize_single_KNN()
        UCFKNN = self.optimize_single_KNN()
        CBFKNN = self.optimize_single_KNN()

        return (ICFKNN, UCFKNN, CBFKNN)

    def optimize_slim(self):
        return {"topK": Integer(5, 1000), "epochs": Integer(20, 1500), "symmetric": Categorical([True, False]),
                      "sgd_mode": Categorical(["sgd", "adagrad", "adam"]),
                      "lambda_i": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                      "lambda_j": Real(low=1e-5, high=1e-2, prior='log-uniform'),
                      "learning_rate": Real(low=1e-4, high=1e-1, prior='log-uniform')}

    def rebuild_slim(self, array):
        return {"topK": array[0], "epochs": array[1], "symmetric": array[2],
                "sgd_mode": array[3], "lambda_i": array[4], "lambda_j": array[5], "learning_rate": array[6]}

    def optimize_puresvd(self):
        return {"num_factors": Integer(5, 1000)}

    def rebuild_puresvd(self, array):
        return {"num_factors": array[0]}

    def optimize_als(self):
        return {"alpha_val": Real(low=0, high=2, prior='uniform'), "n_factors": Integer(5, 1000),
                "regularization": Real(low=1e-4, high=10, prior='log-uniform'), "iterations": Integer(5, 50)}

    def rebuild_als(self, array):
        return {"alpha_val": array[0], "n_factors": array[1], "regularization": array[2], "iterations": array[3]}

    def optimize_p3a(self):
        return {"topK": Integer(5, 800), "alpha": Real(low=0, high=2, prior='uniform'),
                                            "normalize_similarity": Categorical([True, False])}
    def rebuild_p3a(self, array):
        return {"topK": array[0], "alpha": array[1], "normalize_similarity": array[2]}

    def optimize_rp3beta(self):
        return {"topK": Integer(5, 800), "alpha": Real(low=0, high=2, prior='uniform'),
                "beta": Real(low=0, high=2, prior='uniform'), "normalize_similarity": Categorical([True, False])}

    def rebuild_rp3beta(self, array):
        return {"topK": array[0], "alpha": array[1], "beta": array[2], "normalize_similarity": array[3]}


    def evaluate(self, hyp):
        # print("NUMBER OF PARAMETERS ON evaluate():" + str(len(hyp)))

        self.recommender = WeightedHybrid(self.urm_train, self.icm, self.rebuild_single_KNN(hyp[0:7]),
                                          self.rebuild_single_KNN(hyp[7:14]), self.rebuild_single_KNN(hyp[14:21]),
                                          self.rebuild_slim(hyp[21:28]), self.rebuild_puresvd(hyp[28:29]),
                                          None, None, self.rebuild_p3a(hyp[29:32]), self.rebuild_rp3beta(hyp[32:36]),
                                          self.rebuild_weights(hyp[36:]))
        self.recommender.fit()
        result = evaluate_algorithm(self.urm_test, self.recommender, at=10)

        return float(result["MAP"] * (-1))

    def evaluate_single(self, hyp):
        self.recommender = WeightedHybrid(self.urm_train, self.icm, self.rebuild_single_KNN(hyp[0:]),
                                          None, None, None, None, None, None, None, None,
                                          {"icfknn":1})
        self.recommender.fit()
        result = evaluate_algorithm(self.urm_test, self.recommender, at=10)

        return float(result["MAP"] * (-1))

    def run(self):
        self.HYP = {}
        self.HYP["p_icfknn"], self.HYP["p_ucfknn"], self.HYP["p_cbfknn"] = self.optimize_all_KNN()
        self.HYP["p_slimbpr"] = self.optimize_slim()
        self.HYP["p_puresvd"] = self.optimize_puresvd()
        #self.HYP["p_als"] = self.optimize_als()
        self.HYP["p_p3a"] = self.optimize_p3a()
        self.HYP["p_rp3b"] = self.optimize_rp3beta()

        self.HYP["weight"] = self.optimeze_weights()

        self.iterator_to_create_dimension(self.HYP)

        res = gp_minimize(self.evaluate, self.hyperparams_values,
                          n_calls=70,
                          n_random_starts=20,
                          n_points=10000,
                          # noise = 'gaussian',
                          noise=1e-5,
                          acq_func='gp_hedge',
                          acq_optimizer='auto',
                          random_state=None,
                          verbose=True,
                          n_restarts_optimizer=10,
                          xi=0.01,
                          kappa=1.96,
                          x0=None,
                          y0=None,
                          n_jobs=-1)

        self.writer.write_report(str(res), self.report_counter)
        self.create_parameters(res["x"])

    def run_single(self):
        self.HYP["p_icfknn"], _, _ = self.optimize_all_KNN()

        self.iterator_to_create_dimension(self.HYP)

        res = gp_minimize(self.evaluate_single, self.hyperparams_values,
                          n_calls=70,
                          n_random_starts=20,
                          n_points=10000,
                          # noise = 'gaussian',
                          noise=1e-5,
                          acq_func='gp_hedge',
                          acq_optimizer='auto',
                          random_state=None,
                          verbose=True,
                          n_restarts_optimizer=10,
                          xi=0.01,
                          kappa=1.96,
                          x0=None,
                          y0=None,
                          n_jobs=-1)

        self.writer.write_report(str(res), self.report_counter)
        self.create_parameters(res["x"])

    def iterator_to_create_dimension(self, to_iterate):
        skopt_types = [Real, Integer, Categorical]
        for name, hyperparam in to_iterate.items():
            if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
                self.hyperparams_names.append(name)
                self.hyperparams_values.append(hyperparam)
                self.hyperparams[name] = hyperparam

            elif isinstance(hyperparam, str) or isinstance(hyperparam, int) or isinstance(hyperparam, bool):
                self.hyperparams_single_value[name] = hyperparam
            elif isinstance(hyperparam, dict):
                self.iterator_to_create_dimension(to_iterate[name])
            else:
                raise ValueError("Unexpected parameter type: {} - {}".format(str(name), str(hyperparam)))

    def create_parameters(self, hyp):
        self.report_counter = self.report_counter + 1

        self.writer.write_report("p_icfknn :" + str(self.rebuild_single_KNN(hyp[0:7]) ), self.report_counter)
        self.writer.write_report("p_ucfknn :" + str(self.rebuild_single_KNN(hyp[7:14])), self.report_counter)
        self.writer.write_report("p_cbfknn :" + str(self.rebuild_single_KNN(hyp[14:21])), self.report_counter)
        self.writer.write_report("p_slimbpr :" + str(self.rebuild_slim(hyp[21:28])), self.report_counter)
        self.writer.write_report("p_puresvd :" + str(self.rebuild_puresvd(hyp[28:29])), self.report_counter)
        self.writer.write_report("p_p3a :" + str(self.rebuild_p3a(hyp[29:32])), self.report_counter)
        self.writer.write_report("p_rp3b :" + str(self.rebuild_rp3beta(hyp[32:36])), self.report_counter)
        self.writer.write_report("weight :" + str(self.rebuild_weights(hyp[36:])), self.report_counter)


if __name__ == "__main__":
    opt = Optimizer()
    opt.run_single()
