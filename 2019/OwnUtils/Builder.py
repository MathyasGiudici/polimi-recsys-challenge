from OwnUtils.Extractor import Extractor
import pandas as pd
import numpy as np
import scipy.sparse as sps


class Builder(object):

    def __init__(self):

        self.regions = []
        self.ages = []

        self.users_per_region_list = []
        self.users_per_age_list = []

        self.get_regions_and_split_users()
        self.get_ages_and_split_users()

    def build_per_region_urm_all(self):
        """
        Create a list of urm in which users splitting is based on the region they belong to
        """
        urm_per_region_list = []

        for df in self.users_per_region_list:
            row = np.array(df.loc[:, 'row'])
            col = np.array(df.loc[:, 'col'])
            data = np.array(df.loc[:, 'data']).astype(np.int)

            urm_per_region_list.append(sps.coo_matrix((data, (row, col))).tocsr())

        print(urm_per_region_list)

    def build_per_age_urm_all(self):
        """
        Create a list of urm in which users splitting is based on their age
        """
        urm_per_age_list = []

        for df in self.users_per_age_list:
            row = np.array(df.loc[:, 'row'])
            col = np.array(df.loc[:, 'col'])
            data = np.array(df.loc[:, 'data']).astype(np.int)

            urm_per_age_list.append(sps.coo_matrix((data, (row, col))).tocsr())

        print(urm_per_age_list)

    def build_per_region_urm_train(self, urm_train):
        """
        This method constructs the urm_train over a specific region from a csr_matrix.
        It returns a list of csr_matrix, one for each different region.
        """
        if not isinstance(urm_train, sps.csr_matrix):
            raise ValueError("ERROR: works only for CSR format -- use .tocsr() first")

        users_array_list = []

        for df in self.users_per_region_list:
            users = set(df.loc[:, "row"])
            np.argsort(users)
            users = np.array(users).tolist()
            users_array_list.append(users)

        urm_per_region_list = []

        for users_array in users_array_list:
            temp_urm = urm_train.copy()
            print(max(users_array))
            # Create the list of different csr_matrices
            for index in range(0, urm_train.shape[0]):
                if index not in users_array:
                    temp_urm.data[temp_urm.indptr[index] : temp_urm.indptr[index + 1]] = 0

            temp_urm.eliminate_zeros()
            urm_per_region_list.append(sps.csr_matrix(temp_urm))

        return urm_per_region_list


    def build_per_age_urm_train(self, urm_train):
        """
        This method constructs the urm_train over a specific age from a csr_matrix.
        It returns a list of csr_matrix, one for each different age.
        """
        if not isinstance(urm_train, sps.csr_matrix):
            raise ValueError("ERROR: works only for CSR format -- use .tocsr() first")

        users_array_list = []

        for df in self.users_per_age_list:
            users = set(df.loc[:, "row"])
            np.argsort(users)
            users = np.array(users).tolist()
            users_array_list.append(users)

        urm_per_age_list = []

        for users_array in users_array_list:
            temp_urm = urm_train.copy()

            # Create the list of different csr_matrices
            for index in range(0, urm_train.shape[0]):

                if index not in users_array:
                    temp_urm.data[temp_urm.indptr[index] : temp_urm.indptr[index + 1]] = 0

            temp_urm.eliminate_zeros()
            urm_per_age_list.append(sps.csr_matrix(temp_urm))

        return urm_per_age_list


    def get_regions_and_split_users(self):
        """
        Retrieve all the possible typologies of region and a list of sets of users which belong to the same region
        """
        ucm_region_pd = pd.read_csv('C:/Users/david/Documents/PyCharmProjects/RecSys 2019/recsys-challenge/2019/data/data_UCM_region.csv')
        ucm_region = pd.DataFrame(ucm_region_pd)
        self.regions = set(ucm_region.loc[:, "col"])

        for reg in self.regions:
            self.users_per_region_list.append(ucm_region.loc[ucm_region['col'] == reg])


    def get_ages_and_split_users(self):
        """
        Retrieve all the possible typologies of age and a list of sets of users which have the same age
        """
        ucm_age_pd = pd.read_csv('C:/Users/david/Documents/PyCharmProjects/RecSys 2019/recsys-challenge/2019/data/data_UCM_age.csv')
        ucm_age = pd.DataFrame(ucm_age_pd)
        self.ages = set(ucm_age.loc[:, "col"])

        for age in self.ages:
            self.users_per_age_list.append(ucm_age.loc[ucm_age['col'] == age])



# if __name__ == '__main__':
#     builder = Builder()
#     ex = Extractor()
#     urm_train = ex.get_urm_all()
#
#     urm_per_age = builder.build_per_region_urm_train(urm_train)
#     print(urm_per_age[0])
