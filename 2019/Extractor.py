import csv
import numpy as np
import scipy.sparse as sps


class Extractor(object):
    TRAIN_TEST_SPLIT = 0.80
    DATA_FILE_PATH = "data/"

    def get_target_users_of_recs(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "alg_sample_submission.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            for line in csv_reader:
                if line_count != 0:
                    users.append(int(line[0]))
                line_count += 1

            print(f'Processed {line_count} users to make recommendations.')
            return list(set(users))

    def get_interaction_users(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            for line in csv_reader:
                if line_count != 0:
                    users.append(int(line[0]))

                    if int(float(line[2])) != 1:
                        print("Some user has interaction data <= 1")
                line_count += 1

            print(f'Processed {line_count} users from train data.')

            return users

    def get_interaction_items(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[1]))

                    if int(float(line[2])) != 1:
                        print("Some user has interaction data <= 1")
                line_count += 1

            print(f'Processed {line_count} items from train data.')

            return items

    def get_interaction_number(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for _ in csv_reader:
                line_count += 1

            return line_count - 1

    def get_interaction_matrix_all(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            items = []
            for line in csv_reader:
                if line_count != 0:
                    users.append(int(line[0]))
                    items.append(int(line[1]))
                line_count += 1

            print(f'Processed {line_count} interactions.')

            ones_matrix = np.ones(line_count - 1)

            return sps.coo_matrix((ones_matrix, (users, items))).tocsr()

    # Interaction matrix split in test and training
    def get_interaction_matrix_split(self):

        train_mask = np.random.choice([True, False], self.get_interaction_number(self),
                                      p=[self.TRAIN_TEST_SPLIT, 1 - self.TRAIN_TEST_SPLIT])

        user_list = np.array(self.get_interaction_users(self))
        item_list = np.array(self.get_interaction_items(self))
        rating_list = np.ones(self.get_interaction_number(self))

        urm_train = sps.coo_matrix((rating_list[train_mask], (user_list[train_mask], item_list[train_mask])))
        urm_train = urm_train.tocsr()

        test_mask = np.logical_not(train_mask)
        urm_test = sps.coo_matrix((rating_list[test_mask], (user_list[test_mask], item_list[test_mask])))
        urm_test = urm_test.tocsr()

        return [urm_train, urm_test]

    def get_users(self):
        users = self.get_interaction_users(self)
        users.append(self.get_target_users_of_recs(self))

        return list(set(users))

    def get_icm_asset(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_ICM_asset.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            assets = []
            values = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[0]))
                    assets.append(int(line[1]))
                    values.append(float(line[2]))
                line_count += 1

            print(f'Processed {line_count} items.')

            return sps.coo_matrix((values, (items, assets))).tocsr()

    def get_icm_price(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_ICM_price.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            assets = []
            prices = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[0]))
                    assets.append(int(line[1]))
                    prices.append(float(line[2]))
                line_count += 1

            print(f'Processed {line_count} items.')

            return sps.coo_matrix((prices, (items, assets))).tocsr()

    def get_icm_subclass(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_ICM_sub_class.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            assets = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[0]))
                    assets.append(int(line[1]))
                line_count += 1

            print(f'Processed {line_count} items.')

            ones_matrix = np.ones(line_count - 1)

            return sps.coo_matrix((ones_matrix, (items, assets))).tocsr()

    def get_ucm_age(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_UCM_age.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            age_category = []
            for line in csv_reader:
                if line_count != 0:
                    users.append(int(line[0]))
                    age_category.append(int(line[1]))
                line_count += 1

            print(f'Processed {line_count} items.')

            ones_matrix = np.ones(line_count - 1)

            return sps.coo_matrix((ones_matrix, (users, age_category))).tocsr()

    def get_ucm_region(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "data_UCM_region.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            regions = []
            for line in csv_reader:
                if line_count != 0:
                    users.append(int(line[0]))
                    regions.append(int(line[1]))
                line_count += 1

            print(f'Processed {line_count} items.')

            ones_matrix = np.ones(line_count - 1)

            return sps.coo_matrix((ones_matrix, (users, regions))).tocsr()
