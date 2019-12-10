import xlearn as xl


class xLearnRecommender(object):

    def __init__(self, param=None):
        self.path_to_files = "data/xL_data/"
        self.path_to_outs = "./" + self.path_to_files + "out/"

        if param is None:
            self.param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc', 'opt':'sgd'}
        else:
            self.param = param

    def fit(self):
        self.ffm_model = xl.create_ffm()  # Use field-aware factorization machine (ffm)
        self.ffm_model.setTrain(self.path_to_files + "train.txt")  # Path of training data
        self.ffm_model.setTXTModel(self.path_to_files + "model.txt")
        self.ffm_model.fit(self.param, self.path_to_outs + "model.out")
        self.ffm_model.setSigmoid()

    def predict(self):
       self.ffm_model.setTest(self.path_to_files + "validation.txt")
       self.ffm_model.predict(self.path_to_outs + "model.out", self.path_to_outs + "output.txt")

    def test(self):
        self.fit()
        self.predict()