import csv


class Writer(object):
    DATA_FILE_PATH = "data/"

    def write(self, sub_counter, user, rec_list):
        file_name = self.DATA_FILE_PATH + 'subm' + str(sub_counter) + '.csv'
        with open(file_name, mode='a') as csv_file2:
            writer = csv.writer(csv_file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            rec_string = ''

            for el in rec_list:
                rec_string = rec_string + str(el) + " "

            writer.writerow([str(user), rec_string])

    def write_header(self, sub_counter, field_names):
        file_name = self.DATA_FILE_PATH + 'subm' + str(sub_counter) + '.csv'
        with open(file_name, mode='w+') as csv_file2:
            writer = csv.writer(csv_file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(field_names)

        print("Submission file header written")
