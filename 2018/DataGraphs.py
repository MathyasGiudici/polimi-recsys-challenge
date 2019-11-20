import numpy as np
import matplotlib.pyplot as pyplot
from OwnUtils.Extractor import Extractor


def data_visualization():
    # Retriving variables
    userList = list(Extractor().get_users(True))
    itemList = list(Extractor().get_tracks(True, True))
    userList_unique = list(set(userList))
    itemList_unique = list(set(itemList))

    numUsers = len(userList_unique)
    numItems = len(itemList_unique)

    numberInteractions = Extractor().get_numb_interactions()

    print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
    print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
    print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
    print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))

    print("Sparsity {:.2f} %".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))

    URM_all = Extractor().get_train(True)
    URM_all.tocsr()

    itemPopularity = (URM_all > 0).sum(axis=0)
    itemPopularity = np.array(itemPopularity).squeeze()
    pyplot.plot(itemPopularity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Item Index')
    pyplot.show()

    itemPopularity = np.sort(itemPopularity)
    pyplot.plot(itemPopularity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('Item Index')
    pyplot.show()

    userActivity = (URM_all > 0).sum(axis=1)
    userActivity = np.array(userActivity).squeeze()
    userActivity = np.sort(userActivity)

    pyplot.plot(userActivity, 'ro')
    pyplot.ylabel('Num Interactions ')
    pyplot.xlabel('User Index')
    pyplot.show()

if __name__ == '__main__':
    data_visualization()