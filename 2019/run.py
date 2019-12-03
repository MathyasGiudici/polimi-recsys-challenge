from GenericRunner import GenericRunner
from RoundRobinRunner import RoundRobinRunner

if __name__ == '__main__':

    algorithms_choice = {
        "icfknn": True,
        "ucfknn": True,
        "cbfknn": True,
        "slim_bpr": True,
        "pure_svd": True,
        "als": True,
        "cfw": False,
    }

    is_test = True

    #runner = GenericRunner(**algorithms_choice)
    runner = RoundRobinRunner(**algorithms_choice)
    runner.run(is_test)


