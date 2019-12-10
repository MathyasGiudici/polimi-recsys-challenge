import WeightConstants
from OwnUtils.Extractor import Extractor
import Utils.Split.split_train_validation_leave_k_out as loo
from Utils.evaluation_function import evaluate_algorithm
from Hybrid.WeightedHybrid import WeightedHybrid

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

similarity_type = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


class Optimizer(object):

    def __init__(self):
        extractor = Extractor()
        urm = extractor.get_urm_all()
        self.icm = extractor.get_icm_all()

        # Splitting into post-validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

        self.urm_train = matrices[0]
        self.urm_test = matrices[1]

    def optimize_single_KNN(self, obj):
        obj["topK"] = Integer(5, 800)
        obj["shrink"] = Integer(0, 1000)
        obj["similarity"] = Categorical(similarity_type)
        obj["normalize"] = Categorical([True, False])

        if obj["similarity"] == "asymmetric":
            obj["normalize"] = Categorical([True])

        elif obj["similarity"] == "tversky":
            obj["normalize"] = Categorical([True])

        obj["asymmetric_alpha"] = Real(low=0, high=2, prior='uniform')
        obj["tversky_alpha"] = Real(low=0, high=2, prior='uniform')
        obj["tversky_beta"] = Real(low=0, high=2, prior='uniform')

        return obj

    def rebuild_single_KNN(self, array):
        obj = {}
        obj["topK"] = array[0]
        obj["shrink"] = array[1]
        obj["similarity"] = array[2]
        obj["normalize"] = array[3]
        obj["asymmetric_alpha"] = array[4]
        obj["tversky_alpha"] = array[5]
        obj["tversky_beta"] = array[6]
        return obj

    def optimize_all_KNN(self):
        ICFKNN = self.optimize_single_KNN(WeightConstants.ICFKNN)
        UCFKNN = self.optimize_single_KNN(WeightConstants.UCFKNN)
        CBFKNN = self.optimize_single_KNN(WeightConstants.CBFKNN)

        return [ICFKNN, UCFKNN, CBFKNN]

    def optimeze_weights(self):
        # weights = {'icfknn': 2, 'ucfknn': 0.2, 'cbfknn': 0.5, 'slimbpr': 1, 'puresvd': 1.5, 'als': 1, 'cfw': 3, 'p3a': 2, 'rp3b': 3}
        weights = {}
        weights["icfknn"] = Real(low=0, high=5, prior='uniform')
        weights["ucfknn"] = Real(low=0, high=5, prior='uniform')
        weights["cbfknn"] = Real(low=0, high=5, prior='uniform')

        return weights

    def evaluate(self, hyp):
        weights = {}
        weights["icfknn"] = hyp[21]
        weights["ucfknn"] = hyp[22]
        weights["cbfknn"] = hyp[23]
        self.recommender = WeightedHybrid(self.urm_train, self.icm, self.rebuild_single_KNN(hyp[0:7]), self.rebuild_single_KNN(hyp[7:14]), self.rebuild_single_KNN(hyp[14:21]), None, None, None,
                                          None, None, None, weights)
        self.recommender.fit()
        result = evaluate_algorithm(self.urm_test, self.recommender, at=10)

        return float(result["MAP"] * (-1))

    def run(self):
        self.HYP = {}
        p_knns = self.optimize_all_KNN()
        self.HYP["p_icfknn"] = p_knns[0]
        self.HYP["p_ucfknn"] = p_knns[1]
        self.HYP["p_cbfknn"] = p_knns[2]
        self.HYP["weight"] = self.optimeze_weights()

        self.hyperparams = dict()
        self.hyperparams_names = list()
        self.hyperparams_values = list()
        self.hyperparams_single_value = dict()

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
                          y0=None)

        print(res)

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



if __name__ == "__main__":
    opt = Optimizer()
    opt.run()
