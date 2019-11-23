
if __name__ == '__main__':
    from Hybrid.Hybrid import Hybrid

    p_icfknn = {
        "topK": 10,
        "shrink": 10
    }

    p_ucfknn = {
        "topK": 500,
        "shrink": 10
    }

    p_cbfknn = {
        "topK": 100,
        "shrink": 200
    }

    weights = {
        "icfknn" : 0.5,
        "ucfknn" : 0.3,
        "cbfknn" : 0.2,
        "slimbpr" : 0
    }

    isTest = True

    if isTest:
        recommender = Hybrid(isTest, p_icfknn, p_ucfknn, p_cbfknn, None, weights)
        recommender.fit()
        from Utils.evaluation_function import evaluate_algorithm
        evaluate_algorithm(recommender.urm_test, recommender)
    else:
        from OwnUtils.Extractor import Extractor
        from OwnUtils.Writer import Writer

        extractor = Extractor
        users = extractor.get_target_users_of_recs(extractor)

        writer = Writer
        writer.write_header(writer, sub_counter=1)

        recommender = Hybrid(isTest, p_icfknn, p_ucfknn, p_cbfknn, None, weights)
        recommender.fit()

        for user_id in users:
            recs = recommender.recommend(user_id, at=10)
            writer.write(writer, user_id, recs, sub_counter=1)

        print("Submission file written")