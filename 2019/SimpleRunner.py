if __name__ == '__main__':
    from Hybrid.Hybrid import Hybrid
    from OwnUtils.Extractor import Extractor
    from OwnUtils.Writer import Writer


    import Utils.Split.split_train_validation_leave_k_out as loo
    #   from Utils.Split.DataReader_utils import remove_empty_rows_and_cols


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


    isTest = True

    if isTest:
        extractor = Extractor
        urm = extractor.get_interaction_matrix_all(extractor)
        icm = extractor.get_icm_all(extractor)

        # Splitting into validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, True, True)

        urm = matrices[0]
        urm_test = matrices[1]
        urm_validation = matrices[2]

        # Weights of test
        # W = [{
        #     "icfknn": 2,
        #     "ucfknn": 0.7,
        #     "cbfknn": 0.5,
        #     "slimbpr": 1,
        #     "puresvd": 2,
        #     "als": 1,
        # }, {
        #     "icfknn": 2.5,
        #     "ucfknn": 0.2,
        #     "cbfknn": 0.5,
        #     "slimbpr": 1.5,
        #     "puresvd": 2,
        #     "als": 1,
        # }, {
        W = [{
            "icfknn": 2.5,
            "ucfknn": 0.2,
            "cbfknn": 0.5,
            "slimbpr": 1.5,
            "puresvd": 2.3,
            "als": 1.5,
        }]

        # EVALUATION
        results = []

        from Utils.evaluation_function import evaluate_algorithm
        import random

        writer = Writer
        report_counter = 3

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

        generated_weights = []

        for weight in W:
            for _ in range(0,10):
                weight["icfknn"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)
                weight["ucfknn"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)
                weight["cbfknn"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)
                weight["slimbpr"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)
                weight["puresvd"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)
                weight["als"] += round(random.uniform(- max(0.5, weight["icfknn"]), 0.5), 2)

                generated_weights.append(weight.copy())
                print("--------------------------------------")
                recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weight)
                recommender.fit()
                result_dict = evaluate_algorithm(urm_validation, recommender)
                results.append(float(result_dict["MAP"]))

                writer.write_report(writer, str(weight), report_counter)
                writer.write_report(writer, str(result_dict), report_counter)


        # Retriving correct weight
        results.sort()
        weight = generated_weights[int(results.index(max(results)))]

        writer.write_report(writer, "--------------------------------------", report_counter)
        writer.write_report(writer, "TESTING", report_counter)
        writer.write_report(writer, "--------------------------------------", report_counter)

        import scipy.sparse as sps
        # TODO: CHECK IF IT IS CORRECT
        urm = sps.vstack([urm, urm_validation], 'csr')

        recommender = Hybrid(urm, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weight)
        recommender.fit()
        result_dict = evaluate_algorithm(urm_test, recommender)

        writer.write_report(writer, str(weight), report_counter)
        writer.write_report(writer, str(result_dict), report_counter)


    else:
        extractor = Extractor
        users = extractor.get_target_users_of_recs(extractor)
        URM = extractor.get_interaction_matrix_all(extractor)
        ICM = extractor.get_icm_all(extractor)

        writer = Writer
        sub_counter = 1
        writer.write_header(writer, sub_counter=sub_counter)

        recommender = Hybrid(URM, ICM, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, weights)
        recommender.fit()

        from tqdm import tqdm

        for user_id in tqdm(users):
            recs = recommender.recommend(user_id, at=10)
            writer.write(writer, user_id, recs, sub_counter=sub_counter)

        print("Submission file written")
