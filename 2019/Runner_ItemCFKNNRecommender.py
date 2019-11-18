from Extractor import Extractor
from ItemCFKNNRecommender import ItemCFKNNRecommender
from Writer import Writer


if __name__ == '__main__':
    # Function to launch all the others
    extractor = Extractor
    users = extractor.get_target_users_of_recs(extractor)
    URM_all = extractor.get_interaction_matrix_all(extractor)

    writer = Writer
    writer.write_header(writer, sub_counter=1)

    recommender = ItemCFKNNRecommender(URM_all)
    recommender.fit(shrink=50.0, topK=10)

    for user_id in users:
        recs = recommender.recommend(user_id, at=10)
        writer.write(writer, user_id, recs, sub_counter=1)

    print("Submission file written")