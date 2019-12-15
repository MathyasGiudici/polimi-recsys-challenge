from GenericRunner import GenericRunner
from RoundRobinRunner import RoundRobinRunner
from UserFeaturesRunner import UserFeaturesRunner
from CrossValidationRunner import CrossValidationRunner

if __name__ == '__main__':

    algorithms_choice = {
        "icfknn": True,
        "ucfknn": True,
        "cbfknn": True,
        "slim_bpr": True,
        "pure_svd": True,
        "als": True,
        "cfw": False,
        "p3a": False,
        "rp3b": False,
}

    is_test = False
    isSSLIM = False


    runner = GenericRunner(**algorithms_choice)
    # runner = RoundRobinRunner(**algorithms_choice)
    # runner = UserFeaturesRunner(True, True, pure_svd_addition=False, slim_bpr_addition=False)

    # runner = CrossValidationRunner(**algorithms_choice)
    #for i in range(0, 4):
    runner.run(is_test, isSSLIM)


