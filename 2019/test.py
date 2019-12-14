from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
from OwnUtils.Extractor import Extractor
from OwnUtils.Builder import Builder
import pandas as pd
import Utils.Split.split_train_validation_leave_k_out as loo

# if __name__ == '__main__':
#
#     ex = Extractor()
#     urm = ex.get_urm_all()
#     print(urm.shape)
#     icm = ex.get_icm_all()
#     ucm = ex.get_ucm_all()
#
#     matrices = loo.split_train_leave_k_out_user_wise(urm, 1, False, True)
#
#     urm_train = matrices[0]
#     urm_test = matrices[1]
#
#     print("Model fitting...")
#     model = LightFM(loss='warp', learning_schedule='adagrad')
#     model.fit(urm_train, user_features=ucm, item_features=icm, epochs=30, num_threads=4)
#
#     print("Computing precision...")
#     test_precision = precision_at_k(model, urm_test, user_features=ucm, item_features=icm, k=10).mean()
#     test_auc = auc_score(model, urm_test, user_features=ucm, item_features=icm).mean()
#
#     print("Precision is: " + str(test_precision))
#     print("Auc is: " + str(test_auc))


if __name__ == '__main__':
    builder = Builder()
    ex = Extractor()

    list = []
    icm_asset_df = builder.build_icm_asset_dataframe()


    list.extend(icm_asset_df.loc[icm_asset_df["row"] == 18494]["data"].values)
    list.extend(icm_asset_df.loc[icm_asset_df["row"] == 0]["data"].values)
    list.extend(icm_asset_df.loc[icm_asset_df["row"] == 1]["data"].values)

    print(list)

    #builder.split_4_urm()