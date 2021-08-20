
# LoPAD --------
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
from feature_selection import read_mb
from utils import DATA_PATH
from timeit import default_timer as timer
from models.linear_regr import linear_regr
from src.ModelTree import ModelTree
from sklearn.neural_network import MLPRegressor

MB_DATA_PATH = './mb_files'
MB_FILE_NAME = '_training_data_mb.txt'

def get_score_from_dev(test_dev, train_dev, scoring):
    # normalization
    scaler = StandardScaler().fit(train_dev)
    test_dev_norm = scaler.transform(test_dev)
    train_dev_norm = scaler.transform(train_dev)
    if scoring == 'cityblock':
        train_dev_mean = np.mean(train_dev_norm, axis=0).reshape((1, -1))
        test_score = distance.cdist(test_dev_norm, train_dev_mean, 'cityblock').reshape(test_dev.shape[0])
    else:
        test_dev_norm[test_dev_norm < 0] = 0
        test_score = np.sum(test_dev_norm, axis=1)
    return test_score


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


def lopad(data_name, x_train, x_test, y_test, dep_model='tree', scoring='cityblock', dep_rel='all', program_start_time=None):
    clfs = ['lasso', 'tree', 'svm', 'm5p', 'mlp']
    if dep_model not in clfs:
        print(f'the input dependency model({dep_model}) is wrong. Valid: {clfs}')

    test_exp = np.copy(x_test)
    train_exp = np.copy(x_train)

    if dep_rel == 'mb':
        mb = read_mb(f'{MB_DATA_PATH}/{data_name}{MB_FILE_NAME}')
        foc_var = mb['focus']
        mbs = mb['mbs']
        iter_num = len(foc_var)
    else:
        iter_num = x_train.shape[1]

    if dep_model == 'lasso':
        clf = linear_model.Lasso(alpha=0.1)
    elif dep_model == 'tree':
        clf = BaggingRegressor(n_estimators=25, n_jobs=-1)
    elif dep_model == 'svm':
        clf = BaggingRegressor(base_estimator=SVR())
    elif dep_model == 'mlp':
        clf = MLPRegressor(hidden_layer_sizes=500, solver='adam', random_state=1, max_iter=5000, early_stopping=True)
    elif dep_model == 'm5p':
        model = linear_regr()
        clf = ModelTree(model, max_depth=5, min_samples_leaf=7, search_type="greedy", n_search_grid=200)
    else:
        print(f"The specified model is not supported")
        exit(-1)

    train_time_list = []
    test_time_list = []

    for i in range(iter_num):
        train_start = timer()
        idx_predictor = np.delete(np.arange(x_train.shape[1]), [i])
        if dep_rel == 'mb':
            X = x_train[:, mbs[i]].reshape(-1, len(mbs[i]))
            y = x_train[:, foc_var[i]].reshape(-1)
            clf.fit(X, y)
            if i == 0:
                if program_start_time is None:
                    program_start_time = train_start
                train_time_list.append(float(timer() - program_start_time))
            else:
                train_time_list.append(float(timer() - train_start))
            test_start = timer()
            X1 = x_test[:, mbs[i]]
            test_exp[:, foc_var[i]] = clf.predict(X1)
            train_exp[:, foc_var[i]] = clf.predict(X)
            test_time_list.append(float(timer()-test_start))
        else:
            clf.fit(x_train[:, idx_predictor], x_train[:, i])
            if i == 0:
                train_time_list.append(float(timer() - program_start_time))
            else:
                train_time_list.append(float(timer() - train_start))
            test_start = timer()
            test_exp[:, i] = clf.predict(x_test[:, idx_predictor])
            train_exp[:, i] = clf.predict(x_train[:, idx_predictor])
            test_time_list.append(float(timer()-test_start))

    test_dev = abs(x_test - test_exp)
    train_dev = abs(x_train - train_exp)
    test_score = get_score_from_dev(test_dev, train_dev, scoring)

    train_time = np.sum(np.array(train_time_list).reshape(1, len(train_time_list)), axis=1)
    test_time = np.sum(np.array(test_time_list).reshape(1, len(test_time_list)), axis=1)
    return test_score, train_time, test_time
