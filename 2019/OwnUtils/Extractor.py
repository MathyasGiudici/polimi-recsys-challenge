import csv
import numpy as np
import scipy.sparse as sps


class Extractor(object):
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
            return users

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

    def get_interaction_rating(self):
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
