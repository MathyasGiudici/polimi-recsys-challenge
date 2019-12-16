import csv, os



class Writer(object):
    DATA_FILE_PATH = "data/"
    SUBM_FILE_PATH = "submissions/"
    REPORT_FILE_PATH = "reports/"

    def write(self, user, rec_list, sub_counter=0):
        file_name = self.SUBM_FILE_PATH + 'subm' + str(sub_counter) + '.csv'
        with open(file_name, mode='a') as csv_file2:
            writer = csv.writer(csv_file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            rec_string = ''

            for el in rec_list:
                rec_string = rec_string + str(el) + " "

            writer.writerow([str(user), rec_string])

    def write_header(self, sub_counter=0, field_names=['user_id', 'item_list']):
        file_name = self.SUBM_FILE_PATH + 'subm' + str(sub_counter) + '.csv'
        with open(file_name, mode='w+') as csv_file2:
            writer = csv.writer(csv_file2, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(field_names)

        print("Submission file header written")

    def write_report(self, to_write, rep_counter=0):
        file_name = self.REPORT_FILE_PATH + "report" + str(rep_counter) + ".txt"
        f = open(file_name, "a+")
        f.write(to_write + "\n")
        f.close()

    def write_generic(self, file_path, to_write):
        f = open(file_path,"a+")
        f.write(to_write)
        f.close()
