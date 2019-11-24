from Utils.Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np


class UserCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
        self.recs = self.W_sparse.dot(self.URM)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product

        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_expected_ratings(self, user_id):
        expected_ratings = self.recs[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores