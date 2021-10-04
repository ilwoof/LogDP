import os

import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from utils import DATA_PATH, MB_PATH
import re
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# functions
def format_mb(mb_one):
    # mb_one = '1  2 3 4   '
    mb = mb_one.strip()  # strip remove leading and tailing whitespace
    mb = mb.replace("  ", " ")  # replace two whitespace to one whitespace
    mb = mb.split(" ")
    mb = [int(i) for i in mb]
    return mb


def read_mb(filename):
    filename = open(filename, "r")
    mbs_raw = filename.read()
    filename.close()

    # compute how many lines in mbs_raw
    # format in each line: tgt_varaible  mb1 mb2 ... \n
    idx_return = re.finditer('\n', mbs_raw)
    idx_return = [idx.start() for idx in idx_return]

    start = 0
    mbs = []
    focus = []
    for i_mb in range(len(idx_return)):
        # i_mb = 0
        mb_one = mbs_raw[start:idx_return[i_mb]]
        mb_one = format_mb(mb_one)
        # only add to mbs when len>1
        # if len == 1, no mb is founded for the focused variable
        if len(mb_one) > 1:
            focus.append(mb_one[0])
            mbs.append(mb_one[1:])
        start = idx_return[i_mb] + 1  # jump to next line
    return {'focus': focus, 'mbs': mbs}


def lopad_get_exp(data_name, x_trn, x_tst, x_val=None, dep_model='mlp', n_trees=10):
    tst_exp = np.copy(x_tst)
    trn_exp = np.copy(x_trn)
    val_exp = np.copy(x_val) if x_val is not None else None

    # read MB
    # MB could be learned with CausalFS from this paper:
    # Kui Yu, Xianjie Guo, Lin Liu, Jiuyong Li, Hao Wang, Zhaolong Ling, and Xindong Wu. 2020. Causality-Based
    # Feature Selection: Methods and Evaluations. ACM Comput. Surv. 53, 5, Article 111 (Sept. 2020), 36 pages.
    mb_filename = os.path.join(MB_PATH, f'{data_name}_training_data_mb.txt')
    mbs_all = read_mb(mb_filename)
    foc_var = mbs_all['focus']
    mbs = np.array(mbs_all['mbs'])

    # get the features corresponding to mbs
    all0_idx = np.load(os.path.join(DATA_PATH, f'{data_name}_Etrn_all0_idx.npy'))
    mb_features = np.delete(np.arange(x_trn.shape[1]), all0_idx)
    foc_var = mb_features[foc_var]

    for i in range(x_trn.shape[1]):
        if i in foc_var:
            # for dependent events
            if dep_model == 'lasso':
                clf = linear_model.Lasso(alpha=0.1)
            elif dep_model == 'knn':
                clf = KNeighborsRegressor()
            elif dep_model == 'tree':
                clf = BaggingRegressor(base_estimator=DecisionTreeRegressor(ccp_alpha=0.03),
                                       n_estimators=n_trees, n_jobs=-1)
            elif dep_model == 'svm':
                clf = BaggingRegressor(base_estimator=SVR())
            elif dep_model == 'mlp':
                clf = MLPRegressor(max_iter=2000,
                                   hidden_layer_sizes=(50, 50),
                                   #learning_rate_init=0.01,
                                   early_stopping=True,
                                   n_iter_no_change=10)
                #---------------------------------------------------
                # estimator = MLPRegressor(max_iter=5000, n_iter_no_change=10)
                #
                # param_grid = {'hidden_layer_sizes': [(50, ), (200, ), (500,), (50, 50), (100, 100)],
                #               }
                # gsc = GridSearchCV(
                #     estimator,
                #     param_grid,
                #     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
                # idx_predictor = mbs[np.where(foc_var == i)[0]][0]
                # grid_result = gsc.fit(x_trn[:, idx_predictor], x_trn[:, i])
                # best_params = grid_result.best_params_
                #
                # clf = MLPRegressor(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                #                         max_iter=5000, n_iter_no_change=100 )
                #--------------------------------------------------

            idx_predictor = mbs[np.where(foc_var == i)[0]][0]
            clf.fit(x_trn[:, idx_predictor], x_trn[:, i])
            tst_exp[:, i] = clf.predict(x_tst[:, idx_predictor])
            trn_exp[:, i] = clf.predict(x_trn[:, idx_predictor])
            val_exp[:, i] = clf.predict(x_val[:, idx_predictor]) if x_val is not None else None
        else:
            # for independent events
            tst_exp[:, i] = np.mean(x_trn[:, i])
            trn_exp[:, i] = np.mean(x_trn[:, i])
            val_exp[:, i] = np.mean(x_trn[:, i]) if x_val is not None else None
    return trn_exp, tst_exp, val_exp


def get_score_from_dev(test_dev, train_dev):
    # normalization
    scaler = StandardScaler().fit(train_dev)
    dev_norm = scaler.transform(test_dev)
    dev_norm[dev_norm < 0] = 0
    test_score = np.sum(dev_norm, axis=1)

    dev_norm = scaler.transform(train_dev)
    dev_norm[dev_norm < 0] = 0
    train_score = np.sum(dev_norm, axis=1)
    return test_score, train_score


# def get_label_from_scores(scores):
#     y_pred = np.zeros(len(scores))
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(scores.reshape(-1, 1))
#     c1_mean = np.mean(scores[kmeans.labels_ == 0])
#     c2_mean = np.mean(scores[kmeans.labels_ == 1])
#     if c1_mean > c2_mean:
#         y_pred[kmeans.labels_ == 0] = 1
#     else:
#         y_pred[kmeans.labels_ == 1] = 1
#     return y_pred



def lopad(x_train, x_test, dep_model='tree', n_trees=10):
    clfs = ['lasso', 'knn', 'tree', 'svm', 'mlp']
    if dep_model not in clfs:
        print(f'the input dependency model({dep_model}) is wrong. Valid: {clfs}')

    train_exp, test_exp = lopad_get_exp(x_train, x_test, dep_model=dep_model, n_trees=n_trees)

    test_dev = abs(x_test - test_exp)
    train_dev = abs(x_train - train_exp)
    test_score, train_score = get_score_from_dev(test_dev, train_dev)

    return test_score, train_score