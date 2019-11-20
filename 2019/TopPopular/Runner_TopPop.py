from TopPopRecommender import TopPopRecommender
from Writer import Writer
import Extractor as ex
import Evaluator as ev
import Data_manager.Split_functions.split_train_validation_leave_k_out as loo

if __name__ == '__main__':
    # Function to launch all the others

    extractor = ex.Extractor()

    users = extractor.get_target_users_of_recs()

    writer = Writer
    writer.write_header(writer)

    validation_set_needed = True
    matrix = loo.split_train_leave_k_out_user_wise(extractor.get_interaction_matrix_all(), 1, validation_set_needed, True)

    if validation_set_needed:
        urm_train = matrix[0]
        urm_test = matrix[1]
        urm_validation = matrix[2]
        print(matrix)
    else:
        urm_train = matrix[0]
        urm_test = matrix[1]

    topPopRecommender_removeSeen = TopPopRecommender()
    topPopRecommender_removeSeen.fit(urm_train)

    unique_users = list(set(extractor.get_interaction_users()))

    if validation_set_needed:
        ev.evaluate_algorithm(urm_validation, topPopRecommender_removeSeen, unique_users, at=10)
    else:
        ev.evaluate_algorithm(urm_test, topPopRecommender_removeSeen, unique_users, at=10)

    for user_id in users:
        recs = topPopRecommender_removeSeen.recommend(user_id, at=10)
        writer.write(writer,user_id, recs)

    print("Submission file written")