import xlearn as xl
from fastFM import sgd, bpr
from OwnUtils.Extractor import Extractor


class FactorizationMachine(object):
    """
    URM:(30910, 18495)
    """
    def build_matrix(self):
        cols = ['UserID', 'UserRegion', 'UserAge', 'Interactions', 'ItemAsset', 'ItemPrice', 'ItemSubclass']
        extractor = Extractor
        urm_all = extractor.get_urm_all()




