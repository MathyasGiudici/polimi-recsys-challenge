from GenericRunner import GenericRunner

if __name__ == '__main__':

    algorithms_choice = {
        "icfknn": False,
        "ucfknn": False,
        "cbfknn": False,
        "slim_bpr": False,
        "pure_svd": True,
        "als": False,
        "cfw": False,
    }

    is_test = True

    runner = GenericRunner(**algorithms_choice)
    runner.run(is_test)


