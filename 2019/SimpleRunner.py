if __name__ == '__main__':
    from Hybrid.Hybrid import Hybrid
    from OwnUtils.Extractor import Extractor
    from OwnUtils.Writer import Writer
    from datetime import datetime
    from genericRunner import genericRunner
    import scipy.sparse as sp

    import Utils.Split.split_train_validation_leave_k_out as loo
    #   from Utils.Split.DataReader_utils import remove_empty_rows_and_cols

    p_icfknn = {"topK": 10, "shrink": 10}
    p_ucfknn = {"topK": 500, "shrink": 10}
    p_cbfknn = {"topK": 100, "shrink": 200}
    p_slimbpr = {"epochs": 200, "lambda_i": 0.01, "lambda_j": 0.01 }
    p_puresvd = {"num_factors": 1000, }
    p_als = {"alpha_val": 25, "n_factors": 300, "regularization": 0.5, "iterations": 50}
    p_cfw = {"iteration_limit": 50000, "damp_coeff": 0.0, "topK": 300, "add_zeros_quota": 0.0}

    weights = {
        "icfknn": 2.5,
        "ucfknn": 0.2,
        "cbfknn": 0.5,
        "slimbpr": 0.1,
        "puresvd": 2,
        "als": 1,
        "cfw": 3,
    }

    # If isTestis true it  will train the model on the validation and then on the test set
    # otherwise it will write the predictions.
    isTest = True

    if isTest:
        extractor = Extractor
        urm = extractor.get_urm_all(extractor)
        icm = extractor.get_icm_all(extractor)

        # Splitting into post-validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

        urm_post_validation = matrices[0]
        urm_test = matrices[1]

        # Splitting the post-validation matrix in train & validation
        # (Problem of merging train and validation again at the end => loo twice)
        matrices_for_validation = loo.split_train_leave_k_out_user_wise(urm_post_validation, 1, False, True)
        urm_train = matrices_for_validation[0]
        urm_validation = matrices_for_validation[1]

        # Weights of test
        W = [{
            "icfknn": 2.5,
            "ucfknn": 0.2,
            "cbfknn": 0.5,
            "slimbpr": 0.1,
            "puresvd": 2,
            "als": 1,
            "cfw": 3,
        # }, {
        #     "icfknn": 3,
        #     "ucfknn": 0.1,
        #     "cbfknn": 0.5,
        #     "slimbpr": 1.5,
        #     "puresvd": 2,
        #     "als": 0.8,
        # }, {
        #     "icfknn": 3,
        #     "ucfknn": 0.2,
        #     "cbfknn": 0.5,
        #     "slimbpr": 1,
        #     "puresvd": 3,
        #     "als": 1.5,
        }]

        # EVALUATION
        results = []

        from Utils.evaluation_function import evaluate_algorithm
        import random

        writer = Writer
        report_counter = 2
        now = datetime.now()
        date = datetime.fromtimestamp(datetime.timestamp(now))

        writer.write_report(writer, "--------------------------------------", report_counter)
        writer.write_report(writer, "--------------------------------------\n", report_counter)
        writer.write_report(writer, "REPORT " + str(date) + "\n", report_counter)
        writer.write_report(writer, "Fixed parameters", report_counter)
        writer.write_report(writer, str(p_icfknn), report_counter)
        writer.write_report(writer, str(p_ucfknn), report_counter)
        writer.write_report(writer, str(p_cbfknn), report_counter)
        writer.write_report(writer, str(p_slimbpr), report_counter)
        writer.write_report(writer, str(p_puresvd), report_counter)
        writer.write_report(writer, str(p_als), report_counter)

        writer.write_report(writer, str(p_cfw) + "\n", report_counter)

        writer.write_report(writer, "VALIDATION", report_counter)
        writer.write_report(writer, "--------------------------------------", report_counter)

        generated_weights = []

        for weight in W:
            for i in range(0, 1):
                if i != 1:
                    weight["icfknn"] += round(random.uniform(- min(0.5, weight["icfknn"]), 0.5), 2)
                    weight["ucfknn"] += round(random.uniform(- min(0.5, weight["ucfknn"]), 0.5), 2)
                    weight["cbfknn"] += round(random.uniform(- min(0.5, weight["cbfknn"]), 0.5), 2)
                    weight["slimbpr"] += round(random.uniform(- min(0.5, weight["slimbpr"]), 0.5), 2)
                    weight["puresvd"] += round(random.uniform(- min(0.5, weight["puresvd"]), 0.5), 2)
                    weight["als"] += round(random.uniform(- min(0.5, weight["als"]), 0.5), 2)

                    weight["cfw"] += round(random.uniform(- min(0.5, weight["cfw"]), 0.5), 2)


                generated_weights.append(weight.copy())
                print("--------------------------------------")
                recommender = Hybrid(urm_train, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, weight)
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

        recommender = Hybrid(urm_post_validation, icm, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, weight)
        recommender.fit()
        result_dict = evaluate_algorithm(urm_test, recommender)

        writer.write_report(writer, str(weight), report_counter)
        writer.write_report(writer, str(result_dict), report_counter)


    else:
        extractor = Extractor
        users = extractor.get_target_users_of_recs(extractor)
        URM = extractor.get_urm_all(extractor)
        ICM = extractor.get_icm_all(extractor)

        writer = Writer
        sub_counter = 2
        writer.write_header(writer, sub_counter=sub_counter)

        recommender = Hybrid(URM, ICM, p_icfknn, p_ucfknn, p_cbfknn, p_slimbpr, p_puresvd, p_als, p_cfw, weights)
        recommender.fit()

        from tqdm import tqdm

        for user_id in tqdm(users):
            recs = recommender.recommend(user_id, at=10)
            writer.write(writer, user_id, recs, sub_counter=sub_counter)

        print("Submission file written")
