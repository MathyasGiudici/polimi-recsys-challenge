from GenericRunner import GenericRunner
from RoundRobinRunner import RoundRobinRunner
from UserFeaturesRunner import UserFeaturesRunner

if __name__ == '__main__':

    algorithms_choice = {
        "icfknn": True,
        "ucfknn": True,
        "cbfknn": True,
        "slim_bpr": False,
        "pure_svd": False,
        "als": False,
        "cfw": False,
    }

    is_test = True

    #runner = GenericRunner(**algorithms_choice)
    #runner = RoundRobinRunner(**algorithms_choice)
    runner = UserFeaturesRunner(True, True, pure_svd_addition=False, slim_bpr_addition=False)
    runner.run(is_test)


