from GenericRunner import GenericRunner
from RoundRobinRunner import RoundRobinRunner
from UserFeaturesRunner import UserFeaturesRunner

if __name__ == '__main__':

    algorithms_choice = {
        "icfknn": True,
        "ucfknn": True,
        "cbfknn": True,
        "slim_bpr": True,
        "pure_svd": True,
        "als": False,
        "cfw": False,
        "p3a": True,
        "rp3b": True,
    }

    is_test = True

    runner = GenericRunner(**algorithms_choice)
    #runner = RoundRobinRunner(**algorithms_choice)
    #runner = UserFeaturesRunner(True, True, pure_svd_addition=False, slim_bpr_addition=False)
    runner.run(is_test)


