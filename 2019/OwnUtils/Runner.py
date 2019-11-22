from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer

def runner():
    # Getting extractor and urm matrix
    extractor = Extractor()
    users = extractor.get_target_users_of_recs()
    URM_all = extractor.get_interaction_matrix_all()

    # Getting writer
    writer = Writer
    writer.write_header(writer)

    # Getting recommender system
    from CFKNN.ItemCFKNNRecommender import ItemCFKNNRecommender
    recommender = ItemCFKNNRecommender(URM_all)
    recommender.fit(shrink=50.0, topK=10)

    # Making & writing recommendation
    for user_id in users:
        recs = recommender.recommend(user_id, at=10)
        writer.write(writer, user_id, recs)

    print("Submission file written")