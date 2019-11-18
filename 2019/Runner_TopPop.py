from Extractor import Extractor
from TopPopRecommender import TopPopRecommender
from Writer import Writer
import Evaluator as ev

if __name__ == '__main__':
    # Function to launch all the others

    extractor = Extractor
    users = extractor.get_target_users_of_recs(extractor)

    writer = Writer
    writer.write_header(writer)

    matrix = extractor.get_interaction_matrix_split(extractor)
    urm_train = matrix[0]
    urm_test = matrix[1]

    topPopRecommender_removeSeen = TopPopRecommender()
    topPopRecommender_removeSeen.fit(urm_train)

    unique_users = list(set(extractor.get_interaction_users(extractor)))
    ev.evaluate_algorithm(urm_test, topPopRecommender_removeSeen, unique_users, at=10)

    for user_id in users:
        recs = topPopRecommender_removeSeen.recommend(user_id, at=10)
        writer.write(writer,user_id, recs)

    print("Submission file written")