from OwnUtils.Extractor import Extractor
from TopPopRecommender import TopPopRecommender
from OwnUtils.Writer import Writer
import OwnUtils.Evaluator as ev
import numpy as np

if __name__ == '__main__':
    # Function to launch all the others

    SUBMISSION_NUMBER = 1 # TO BE CHANGED MANUALLY
    field_names = ['playlist_id', 'track_ids']
    users = Extractor.get_user_to_make_rec(Extractor)

    Writer.write_header(Writer, SUBMISSION_NUMBER, field_names)

    matricies = Extractor.get_train_test_matrix(Extractor)
    URM_train = matricies[0]
    URM_test = matricies[1]

    topPopRecommender_removeSeen = TopPopRecommender()
    topPopRecommender_removeSeen.fit(URM_train)

    unique_users = list(set(Extractor.get_interaction_users(Extractor, False)))
    ev.evaluate_algorithm(URM_test, topPopRecommender_removeSeen, unique_users, at=10)

    for user_id in users:
        recs = topPopRecommender_removeSeen.recommend(user_id, at=10)
        Writer.write(Writer, SUBMISSION_NUMBER, user_id, recs)

    print("Submission file written")