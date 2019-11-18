# Return how many users are not in train set, but must be recommended as well
if __name__ == '__main__':

    from Extractor import Extractor

    extractor = Extractor

    to_predict = extractor.get_target_users_of_recs(extractor)
    from_train = extractor.get_interaction_users(extractor)

    missing = []

    for el in to_predict:
        if not(el in from_train):
            missing.append(el)

    print("Users to make prediction that are not in train:" + str(len(missing)))