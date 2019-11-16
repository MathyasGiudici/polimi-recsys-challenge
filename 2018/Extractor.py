import csv
import numpy as np
import scipy.sparse as sps

class Extractor(object):
    train_test_split = 0.80
    DATA_FILE_PATH = "data/"

    def get_users(self, debugMode):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "target_playlists.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            users = []
            for row in csv_reader:
                if line_count == 0:
                    if debugMode:
                        print(f'Column names are {", ".join(row)}')
                elif line_count < 10:
                    if debugMode:
                        print(f'{row[0]}')
                    users.append(int(row[0]))
                else:
                    users.append(int(row[0]))
                line_count += 1

            print(f'Processed {line_count} users.')

            return users

    def get_interaction_matrix(self, debugMode):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            rows = []
            cols = []
            for row in csv_reader:
                if line_count == 0:
                    if debugMode:
                        print(f'Column names are {", ".join(row)}')
                elif line_count < 10:
                    if debugMode:
                        print(f'{row[0]}, {row[1]}')
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                else:
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                line_count += 1

            print(f'Processed {line_count} interactions.')

            ones_matrix = np.ones(line_count - 1)

            return sps.coo_matrix((ones_matrix, (rows, cols))).tocsr()

    def get_interaction_users(self, debugMode):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            rows = []
            cols = []
            for row in csv_reader:
                if line_count == 0:
                    if debugMode:
                        print(f'Column names are {", ".join(row)}')
                elif line_count < 10:
                    if debugMode:
                        print(f'{row[0]}, {row[1]}')
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                else:
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                line_count += 1

            print(f'Processed {line_count} users from train.')

            return rows

    def get_interaction_items(self, debugMode):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            rows = []
            cols = []

            for row in csv_reader:
                if line_count == 0:
                    if debugMode:
                        print(f'Column names are {", ".join(row)}')
                elif line_count < 10:
                    if debugMode:
                        print(f'{row[0]}, {row[1]}')
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                else:
                    rows.append(int(row[0]))
                    cols.append(int(row[1]))
                line_count += 1

            print(f'Processed {line_count} tracks from train.')

            return cols

    def get_numb_interactions(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "train.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                line_count += 1

            return line_count - 1


    def get_train_test_matrix(self):

        train_mask = np.random.choice([True, False], Extractor().get_numb_interactions(),
                                      p=[self.train_test_split, 1 - self.train_test_split])

        userList = np.array(self.get_interaction_users(self, False))
        itemList = np.array(self.get_interaction_items(self, False))
        ratingList = np.ones(Extractor().get_numb_interactions())

        URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
        URM_train = URM_train.tocsr()

        test_mask = np.logical_not(train_mask)
        URM_test = sps.coo_matrix((ratingList[test_mask], (userList[test_mask], itemList[test_mask])))
        URM_test = URM_test.tocsr()

        return [URM_train, URM_test]

    def get_tracks(self, onlyIds, debugMode):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "tracks.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            tracks = []
            tracks_id = []
            for row in csv_reader:
                if line_count == 0:
                    if debugMode:
                        print(f'Column names are {", ".join(row)}')
                elif line_count < 10:
                    if debugMode:
                        print(f'{row[0]}, {row[1]}, {row[2]}, {row[3]}')
                    tracks_id.append(int(row[0]))
                    tracks.append(tuple([int(row[0]), int(row[1]), int(row[2]), int(row[3])]))
                else:
                    tracks_id.append(int(row[0]))
                    tracks.append(tuple([int(row[0]), int(row[1]), int(row[2]), int(row[3])]))
                line_count += 1

            print(f'Processed {line_count} items.')

            if onlyIds:
                return tracks_id
            else:
                return tracks

    def get_user_to_make_rec(self):
        # Composing the name
        file_name = self.DATA_FILE_PATH + "sample_submission.csv"

        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            fieldnames = []
            users = []
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    fieldnames.append(row[0])
                    fieldnames.append(row[1])
                else:
                    users.append(int(row[0]))
                line_count += 1

            print(f'Processed {line_count} users to make recommendation.')
            return users
