import numpy as np
from Notebooks_utils.data_splitter import train_test_holdout
from Notebooks_utils.evaluation_function import evaluate_algorithm
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import scipy.sparse as sps
from Extractor import Extractor
from Writer import Writer


class UserCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product

        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

if __name__ == '__main__':
    extractor = Extractor
    userList = extractor.get_interaction_users(extractor, False)
    itemList = extractor.get_interaction_items(extractor, False)
    ratingList = np.ones(Extractor().get_numb_interactions())

    URM_all = extractor.get_interaction_matrix(extractor, False)
    warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > 0
    warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

    URM_all = URM_all[:, warm_items]

    warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > 0
    warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

    URM_all = URM_all[warm_users, :]

    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

    recommender = UserCFKNNRecommender(URM_train)
    recommender.fit(shrink=0.0, topK=200)

    submissionID = 3

    userList_unique = extractor.get_user_to_make_rec(extractor)

    writer = Writer()
    fields = ['playlist_id', 'track_ids']
    writer.write_header(submissionID, fields)

    for user_id in userList_unique:
        writer.write(submissionID, user_id, recommender.recommend(user_id, at=10))
    print("Done")

