import numpy as np
import scipy.sparse as sps
from sklearn.model_selection import LeaveOneOut

class Splitter(object):
    TRAIN_TEST_SPLIT = 0.80
    RANDOM = "Random"
    LOO = "LeaveOneOut"
    VALIDATION = "Validation"

    # Split of train and test randomly with TRAIN_TEST_SPLIT parameter
    def random_split(self, ex):
        train_mask = np.random.choice([True, False], ex.get_interaction_rating(),
                                      p=[self.TRAIN_TEST_SPLIT, 1 - self.TRAIN_TEST_SPLIT])

        user_list = np.array(ex.get_interaction_users())
        item_list = np.array(ex.get_interaction_items())
        rating_list = np.ones(ex.get_interaction_rating())

        urm_train = sps.coo_matrix((rating_list[train_mask], (user_list[train_mask], item_list[train_mask])))
        urm_train = urm_train.tocsr()

        test_mask = np.logical_not(train_mask)
        urm_test = sps.coo_matrix((rating_list[test_mask], (user_list[test_mask], item_list[test_mask])))
        urm_test = urm_test.tocsr()

        return [urm_train, urm_test, None]

    # Split of train and test with Leave One Out method
    def leave_one_out_split(self, ex):
        user_list = np.array(ex.get_interaction_users())
        item_list = np.array(ex.get_interaction_items())
        rating_list = np.ones(ex.get_interaction_rating())

        urm_all = ex.get_interaction_matrix_all()

        train_mask = []

        for user in list(set(user_list)):
            numItems = len(urm_all.getrow(user).indices)
            true_matrix = np.ones(numItems, dtype="bool")
            if numItems > 1:
                true_matrix[np.random.randint(0, numItems)] = False
            train_mask.append(true_matrix)

        print(train_mask)

        train_mask = np.array(train_mask)
        urm_train = sps.coo_matrix((rating_list[train_mask], (user_list[train_mask], item_list[train_mask])))
        urm_train = urm_train.tocsr()

        test_mask = np.logical_not(train_mask)
        urm_test = sps.coo_matrix((rating_list[test_mask], (user_list[test_mask], item_list[test_mask])))
        urm_test = urm_test.tocsr()

        return [urm_train, urm_test, None]


    # Choose the wanted method to split data
    def choose_split_type(self, ex, type_of_split):
        if type_of_split == self.RANDOM:
            return self.random_split(ex)

        elif type_of_split == self.LOO:
            return self.leave_one_out_split(ex)

        else:
            print("---> Error in selecting splitting method!")