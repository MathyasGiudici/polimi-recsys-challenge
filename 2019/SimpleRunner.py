if __name__ == '__main__':
    from Hybrid.Hybrid import Hybrid
    from OwnUtils.Extractor import Extractor
    from OwnUtils.Writer import Writer


    import Utils.Split.split_train_validation_leave_k_out as loo
    from Utils.Split.DataReader_utils import remove_empty_rows_and_cols


    p_icfknn = {"topK": 10, "shrink": 10}

    p_ucfknn = {"topK": 500, "shrink": 10}

    p_cbfknn = {"topK": 100, "shrink": 200}

    p_slimbpr = {"epochs": 200, "lambda_i": 0.01, "lambda_j": 0.01 }

    p_puresvd = {"num_factors": 1000, }

    p_als = {"alpha_val": 25, "n_factors": 300, "regularization": 0.5, "iterations": 50}

    weights = {
        "icfknn": 2.5,
        "ucfknn": 0.2,
        "cbfknn": 0.5,
        "slimbpr": 1.5,
        "puresvd": 2,
        "als": 1,
    }


    isTest = False

    if isTest:
        extractor = Extractor
        urm = extractor.get_interaction_matrix_all(extractor)
        icm = extractor.get_icm_all(extractor)

        # TODO: maybe not to remove to have also cold items and users
        # matrices = remove_empty_rows_and_cols(urm, icm)
        # urm = matrices[0]
        # icm = matrices[1]

        # Splitting into validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, True, True)

        urm = matrices[0]
        urm_test = matrices[1]
        urm_validation = matrices[2]

        # Weights of test
        W = [{
            "icfknn": 0.5,
            "ucfknn": 0.3,
            "cbfknn": 0.2,
            "slimbpr": 0.1,
            "puresvd": 0.4,
            "als": 0.2,
        }, {
            "icfknn": 0.4,
            "ucfknn": 0.4,
            "cbfknn": 0.3,
            "slimbpr": 0.2,
            "puresvd": 0.3,
            "als": 0.4,
        }, {
            "icfknn": 0.7,
            "ucfknn": 0.5,
            "cbfknn": 0.5,
            "slimbpr": 0.5,
            "puresvd": 0.5,
            "als": 0.7,
        }, {
            "icfknn": 0.5,
            "ucfknn": 0.2,
            "cbfknn": 0.1,
            "slimbpr": 0.5,
            "puresvd": 1,
            "als": 1,
        }, {
            "icfknn": 0.3,
            "ucfknn": 0.5,
            "cbfknn": 0.5,
            "slimbpr": 0.3,
            "puresvd": 2,
            "als": 1,
        }, {
            "icfknn": 1,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 2,
            "puresvd": 1,
            "als": 2,
        }, {
            "icfknn": 2,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 1,
            "puresvd": 2,
            "als": 1,
        }, {
            "icfknn": 1.5,
            "ucfknn": 0.7,
            "cbfknn": 0.5,
            "slimbpr": 1.5,
            "puresvd": 2,
            "als": 1,
        }, {
            "icfknn": 5,
            "ucfknn": 3,
            "cbfknn": 2,
            "slimbpr": 2.5,
            "puresvd": 4,
            "als": 3,
        }, {
            "icfknn": 0.5,
            "ucfknn": 0.2,
            "cbfknn": 0.7,
            "slimbpr": 1,
            "puresvd": 2,
            "als": 1.5,
        }, {
            "icfknn": 1,
            "ucfknn": 0.2,
            "cbfknn": 0.5,
            "slimbpr": 2,
            "puresvd": 2.5,
            "als": 1,
        }, {
            "icfknn": 2.5,
            "ucfknn": 0.2,
            "cbfknn": 0.5,
            "slimbpr": 1.5,
            "puresvd": 2,
            "als": 1,
        }]


        # EVALUATION
        results = []

        from Utils.evaluation_function import evaluate_algorithm
        writer = Writer
        report_counter = 1

        writer.write_report(writer, "REPORT",report_counter)
        writer.write_report(writer, "Fixed parameters", report_counter)
        writer.write_report(writer, str(p_icfknn), report_counter)
        writer.write_report(writer, str(p_ucfknn), report_counter)
        writer.write_report(writer, str(p_cbfknn), report_counter)
        writer.write_report(writer, str(p_slimbpr), report_counter)
        writer.write_report(writer, str(p_puresvd), report_counter)
        writer.write_report(writer, str(p_als), report_counter)
        writer.write_report(writer, "VALIDATION", report_counter)
        writer.write_report(writer, "--------------------------------------", report_counter)


        for weight in W:
            print("--------------------------------------")
            recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weight)
            recommender.fit()
            result_dict = evaluate_algorithm(urm_validation, recommender)
            results.append(float(result_dict["MAP"]))

            writer.write_report(writer, str(weight), report_counter)
            writer.write_report(writer, str(result_dict), report_counter)


        # Retriving correct weight
        results.sort()
        weight = W[int(results.index(max(results)))]

        writer.write_report(writer, "--------------------------------------", report_counter)
        writer.write_report(writer, "TESTING", report_counter)
        writer.write_report(writer, "--------------------------------------", report_counter)

        import scipy.sparse as sps
        urm = sps.vstack([urm, urm_validation], 'csr')

        recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weight)
        recommender.fit()
        result_dict = evaluate_algorithm(urm_test, recommender)

        writer.write_report(writer, str(weight), report_counter)
        writer.write_report(writer, str(result_dict), report_counter)


    else:
        weights = {
            "icfknn": 1,
            "ucfknn": 0.2,
            "cbfknn": 0.5,
            "slimbpr": 2.5
        }

        extractor = Extractor
        users = extractor.get_target_users_of_recs(extractor)
        URM = extractor.get_interaction_matrix_all(extractor)
        ICM = extractor.get_icm_all(extractor)

        writer = Writer
        writer.write_header(writer, sub_counter=0)

        recommender = Hybrid(URM, ICM, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weights)
        recommender.fit()

        for user_id in users:
            recs = recommender.recommend(user_id, at=10)
            writer.write(writer, user_id, recs, sub_counter=0)

        print("Submission file written")
