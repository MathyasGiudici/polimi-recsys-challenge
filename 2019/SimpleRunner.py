if __name__ == '__main__':
    from Hybrid.Hybrid import Hybrid
    from OwnUtils.Extractor import Extractor
    import Utils.Split.split_train_validation_leave_k_out as loo
    from Utils.Split.DataReader_utils import remove_empty_rows_and_cols

    p_icfknn = {"topK": 10, "shrink": 10}

    p_ucfknn = {"topK": 500, "shrink": 10}

    p_cbfknn = {"topK": 100, "shrink": 200}

    p_slimbpr = {"epochs": 2000,}

    weights = {
        "icfknn": 0.5,
        "ucfknn": 0.3,
        "cbfknn": 0.2,
        "slimbpr": 0.1
    }

    isTest = False

    extractor = Extractor
    urm = extractor.get_interaction_matrix_all(extractor)
    icm = extractor.get_icm_all(extractor)

    if isTest:
        # Splitting into validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, True, True)

        urm = matrices[0]
        urm_test = matrices[1]
        urm_validation = matrices[2]

        W = [{
            "icfknn": 0.5,
            "ucfknn": 0.3,
            "cbfknn": 0.2,
            "slimbpr": 0.1
        }, {
            "icfknn": 0.4,
            "ucfknn": 0.4,
            "cbfknn": 0.3,
            "slimbpr": 0.2
        }, {
            "icfknn": 0.7,
            "ucfknn": 0.5,
            "cbfknn": 0.5,
            "slimbpr": 0.5
        }, {
            "icfknn": 0.5,
            "ucfknn": 0.2,
            "cbfknn": 0.1,
            "slimbpr": 0.5
        }, {
            "icfknn": 0.3,
            "ucfknn": 0.5,
            "cbfknn": 0.5,
            "slimbpr": 0.3
        }, {
            "icfknn": 1,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 2
        }, {
            "icfknn": 2,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 1
        }]

        results = []

        from Utils.evaluation_function import evaluate_algorithm


        for weight in W:
            print("--------------------------------------")
            #self,  urm, icm , p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, weights):
            recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, weight)
            recommender.fit()
            result_dict = evaluate_algorithm(urm_validation, recommender)
            results.append(result_dict["MAP"])

            print(weight)
            print(float(result_dict))

        results.sort()
        print("--------------------------------------")
        print("--------------------------------------")
        for i in range(0,len(results)):
            print("INDEX:",str(i)," MAP:",str(results[i]))

        weight = W[W.index(max(results))]




    else:
        from OwnUtils.Extractor import Extractor
        from OwnUtils.Writer import Writer

        weights = {
            "icfknn": 2,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 1
        }

        extractor = Extractor
        users = extractor.get_target_users_of_recs(extractor)

        writer = Writer
        writer.write_header(writer, sub_counter=3)

        recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, weights)
        recommender.fit()

        for user_id in users:
            recs = recommender.recommend(user_id, at=10)
            writer.write(writer, user_id, recs, sub_counter=3)

        print("Submission file written")
