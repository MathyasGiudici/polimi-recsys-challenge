from OwnUtils.Extractor import Extractor
from OwnUtils.Writer import Writer
from XGBoostFeatures import FeatureAdder
import csv
import pandas as pd

ARGS_FEATURES = {"user_profile_length": True,
                 "top_pop": True}


class XGBoostDataframe(object):
    """
    This is the class used to build a partial dataframe for XGBoost or LightGBM that has to be filled with the runtime
    prediction and with items attributes, which can be known only at runtime
    """
    DATA_FILE_PATH = "data/"
    PREDICTION_FILE = "predictions@"
    TEST_FILE = "xgb_df_test@"

    def __init__(self, group_length):
        self.group_length = group_length

        extractor = Extractor()
        self.users = extractor.get_target_users_of_recs()

    # BUILD USER AND ITEMS LISTS FROM CSV FILE OR RUNTIME
    def get_user_and_item_lists(self):
        file_name = self.DATA_FILE_PATH + self.PREDICTION_FILE + str(self.group_length) + ".csv"
        print("Building dataframe from " + self.PREDICTION_FILE + str(self.group_length) + "...")

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            items = []
            for line in csv_reader:
                if line_count != 0:
                    if line is not None:
                        users.extend([line[0]] * self.group_length)
                        item = line[1].split()
                        items.extend(item)
                line_count += 1

            print("Building completed!")
            return users, items

    # BUILD THE BASE DATAFRAME MADE UP OF ONLY USER AND ITEM COLUMNS
    def build_base_dataframe(self, users, items):
        dataframe = pd.DataFrame({"user_id": users, "item_id": items})
        return dataframe

    # PROXY OF THE REAL METHOD WITHOUT ARGS_FEATURES
    def build_whole_dataframe(self, dataframe):
        self.add_features(dataframe, **ARGS_FEATURES)

    # ADD THE CHOSEN FEATURES TO THE DATAFRAME
    def add_features(self, dataframe,
                     user_profile_length=False,
                     user_region1=False,
                     user_region2=False,
                     user_age=False,
                     top_pop=False,
                     item_asset=False,
                     item_price=False,
                     item_subclass=False,
                     ):

        features_adder = FeatureAdder(dataframe, self.group_length)

        if user_profile_length:
            features_adder.add_user_profile_length()
        if user_region1:
            features_adder.add_user_region1()
        if user_age:
            features_adder.add_user_age()
        if top_pop:
            features_adder.add_item_popularity()

    def retrieve_test_dataframe(self):
        file_name = self.DATA_FILE_PATH + self.TEST_FILE + str(self.group_length) + ".csv"
        return pd.read_csv(file_name)



if __name__ == '__main__':
    xgb = XGBoostDataframe(20)
    users, items = xgb.get_user_and_item_lists()
    dataframe = xgb.build_base_dataframe(users, items)
    xgb.add_features(dataframe, **ARGS_FEATURES)

    writer = Writer()
    # writer.save_dataframe(dataframe, 20)
    print(dataframe)


