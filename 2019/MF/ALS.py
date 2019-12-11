"""
ALS imported from  https://github.com/APPiccio/recsys-polimi-2018
"""

import numpy as np
import implicit


class AlternatingLeastSquare:
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    def __init__(self, URM_train):
        self.URM = URM_train

    def fit(self, alpha_val=25, n_factors=300, regularization=0.15, iterations=50):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha_val = alpha_val

        sparse_item_user = self.URM.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)


        #alpha_val = 1
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * self.alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors


    def get_expected_ratings(self, playlist_id):
        scores = np.dot(self.user_factors[playlist_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]
