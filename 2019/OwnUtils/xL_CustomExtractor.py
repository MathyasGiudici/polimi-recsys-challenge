import csv
import numpy as np
import scipy.sparse as sps

from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer


class CustomExtractor:

    def __init__(self):
        self.extractor = Extractor
        self.writer = Writer()
        self.my_path = self.extractor.DATA_FILE_PATH + "xL_data/prova.txt"

        # Some useful variables
        self.urm = self.extractor.get_urm_all(self.extractor)

        self.ucm_age = self.extractor.get_ucm_age(self.extractor).tocsr()
        self.ucm_region = self.extractor.get_ucm_region(self.extractor).tocsr()

        self.icm_asset = self.get_icm_asset()
        self.icm_price = self.get_icm_price()
        self.icm_sub_cat = self.extractor.get_icm_subclass(self.extractor).tocsr()

    def create_validation_test_files(self, write_userf, write_itemf):

        import Utils.Split.split_train_validation_leave_k_out as loo
        urm = self.urm

        # Splitting into post-validation & testing in case of parameter tuning
        matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)

        urm_post_validation = matrices[0]
        self.my_path = self.extractor.DATA_FILE_PATH + "xL_data/post_validation.txt"
        self.urm = urm_post_validation
        self.create_general_file(write_userf, write_itemf)

        urm_test = matrices[1]
        self.my_path = self.extractor.DATA_FILE_PATH + "xL_data/test.txt"
        self.urm = urm_test
        self.create_general_file(write_userf, write_itemf)

        # Splitting the post-validation matrix in train & validation
        # (Problem of merging train and validation again at the end => loo twice)
        matrices_for_validation = loo.split_train_leave_k_out_user_wise(self.urm_post_validation, 1, False, True)

        urm_train = matrices_for_validation[0]
        self.my_path = self.extractor.DATA_FILE_PATH + "xL_data/train.txt"
        self.urm = urm_train
        self.create_general_file(write_userf, write_itemf)

        urm_validation = matrices_for_validation[1]
        self.my_path = self.extractor.DATA_FILE_PATH + "xL_data/validation.txt"
        self.urm = urm_validation
        self.create_general_file(write_userf, write_itemf)


    def create_general_file(self, write_userf, write_itemf):

        n_users, n_items = self.urm.shape

        from tqdm import tqdm

        for user in tqdm(range(0, n_users)):
            # Getting positive items
            positive_interactions = list(self.urm[user].indices)

            # Checking
            to_create = len(positive_interactions)
            if len(positive_interactions) == 0:
                to_create = 1

            # Generating equal number of negative ones
            negative_interactions = []
            for _ in range(0, to_create):
                value = np.random.randint(0, high=(n_items + 1), size=1)[0]
                while value in positive_interactions:
                    value = np.random.randint(0, high=(n_items + 1), size=1)[0]
                negative_interactions.append(value)

            '''
            File written:
            0/1 if is positive interaction or not
            0 --------- Interaction Layer
                0 - User
                1 - Item
            1 --------- User Content Layer
                0 - Age
                1 - Region 1
                2 - Region 2
            2 --------- Item Content Layer
                0 - Asset
                1 - Price
                2 - Sub Cat
            '''
            # Writing stage
            for item in positive_interactions:
                # 0 --------- Interaction Layer
                row_to_write = "1 0:0:" + str(user) + " 0:1:" + str(item)

                # 1 --------- User Content Layer
                if write_userf:
                    row_to_write += self.create_user_features(user)

                # 2 --------- Item Content Layer
                if write_itemf:
                    row_to_write += self.create_item_features(item)

                row_to_write += "\n"
                self.writer.write_generic(self.my_path, row_to_write)

            for item in negative_interactions:
                # 0 --------- Interaction Layer
                row_to_write = "0 0:0:" + str(user) + " 0:1:" + str(item)

                # 1 --------- User Content Layer
                if write_userf:
                    row_to_write += self.create_user_features(user)

                # 2 --------- Item Content Layer
                if write_itemf:
                    row_to_write += self.create_item_features(item)

                row_to_write += "\n"
                self.writer.write_generic(self.my_path, row_to_write)

    def create_user_features(self, user):
        to_write = ""

        if not user >= self.ucm_age.shape[0]:
            for i in range(0, len(self.ucm_age[user].indices)):
                age = self.ucm_age[user].indices[i]
                to_write += " 1:0:" + str(age)

        if not user >= self.ucm_region.shape[0]:
            for i in range(0, len(self.ucm_region[user].indices)):
                region = self.ucm_region[user].indices[i]
                to_write += " 1:" + str(i+1) + ":" + str(region)

        return to_write

    def create_item_features(self, item):
        to_write = ""

        if not item >= self.icm_asset.shape[0]:
            for i in range(0, len(self.icm_asset[item].data)):
                data = self.icm_asset[item].data[i]
                to_write += " 2:0:" + str(data)

        if not item >= self.icm_price.shape[0]:
            for i in range(0, len(self.icm_price[item].data)):
                data = self.icm_price[item].data[i]
                to_write += " 2:1:" + str(data)

        if not item >= self.icm_sub_cat.shape[0]:
            for i in range(0, len(self.icm_sub_cat[item].indices)):
                data = self.icm_sub_cat[item].indices[i]
                to_write += " 2:2:" + str(data)

        return to_write

    def get_icm_asset(self):
        # Composing the name
        file_name = self.extractor.DATA_FILE_PATH + "data_ICM_asset.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            assets = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[0]))
                    assets.append(float(line[2]))
                line_count += 1

            cols = np.zeros(line_count - 1, dtype=int)

            return sps.coo_matrix((assets, (items, cols))).tocsr()

    def get_icm_price(self):
        # Composing the name
        file_name = self.extractor.DATA_FILE_PATH + "data_ICM_price.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            items = []
            prices = []
            for line in csv_reader:
                if line_count != 0:
                    items.append(int(line[0]))
                    prices.append(float(line[2]))
                line_count += 1

            cols = np.zeros(line_count - 1, dtype=int)

            return sps.coo_matrix((prices, (items, cols))).tocsr()