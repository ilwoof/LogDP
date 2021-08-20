
import os
import pickle
import random
import numpy as np
import sys
sys.path.append('../')
from loglizer import dataloader, preprocessing
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

# <editor-fold desc="------ setting ------">
if os.path.basename(os.getcwd()) == 'LogDependency':
    os.chdir(os.getcwd() + '/src')
DATA_PATH = os.path.dirname(os.getcwd()) + '/data'

RST_HEADER = np.array(['data_name', 'feature_extract', 'feature_type',
                       'n_train_obj', 'n_train_nor', 'n_train_ano', 'n_train_feature',
                       'n_test_obj', 'n_test_nor', 'n_test_ano', 'n_test_feature',
                       'method', 'dep_rel', 'dep_scoring', 'n_feature',
                       'n_report', 'TP', 'FP', 'FN', 'roc_auc', 'pr_auc', 'precision', 'recall', 'f1'])
# </editor-fold>

def evaluate(y_true, score = None, n_report = None, y_pred = None, verbose=1):
    # n_report is only valid when score is inputted
    if score is None and y_pred is None:
        print(f'Need to input score and/or y_pred.')
        return
    if score is not None and n_report is None:
        n_report = np.sum(y_true == 1)

    roc_auc = roc_auc_score(y_true, score)
    prs, res, thresholds = precision_recall_curve(y_true, score)
    # calculate precision-recall AUC
    pr_auc = auc(res, prs)
    if y_pred is None:
        y_pred = np.zeros(len(y_true))
        y_pred[np.argsort(-score)[:n_report]] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    if verbose == 1:
        print(f'n_report={n_report}/{np.sum(y_true==1)}    '
              f'roc_auc={roc_auc:.3f}    pr_auc={pr_auc:.3f}    '
              f'TP={TP}    FP={FP}    FN={FN}    '
              f'precision={precision:.3f}   recall={recall:.3f}   f1={f1:.3f}')
    return TP, FP, FN, round(roc_auc, 3), round(pr_auc, 3), round(precision, 3), round(recall, 3), round(f1, 3)


def load_data(data_name, feature_extract, feature_type):
    x_test, y_test = pickle.load(open(os.path.join(DATA_PATH, data_name + '_test.pkl'), 'rb'))
    x_test = np.asarray(x_test)
    y_test = np.array(y_test).flatten()
    x_train, y_train = pickle.load(open(os.path.join(DATA_PATH, data_name + '_train.pkl'), 'rb'))
    x_train = np.asarray(x_train)
    y_train = np.array(y_train).flatten()
    x_all = np.concatenate((x_train, x_test), axis=None)

    # feature extract
    feature_extractor = preprocessing.FeatureExtractor()
    if feature_extract == 'trn':
        from_data = x_train
    elif feature_extract == 'test':
        from_data = x_test
    else:
        from_data = x_all

    if feature_type == 'raw':
        feature_extractor.fit_transform(from_data)
    elif feature_type == 'wn':
        feature_extractor.fit_transform(from_data, term_weighting='tf-idf', normalization='zero-mean')
    elif feature_type == 'w':
        feature_extractor.fit_transform(from_data, term_weighting='tf-idf')
    elif feature_type == 'n':
        feature_extractor.fit_transform(from_data, normalization='zero-mean')
    else:
        print(f'Wrong feature_type({feature_type}), valid: raw, wn, w, n.')
        return

    x_train = feature_extractor.transform(x_train)
    x_test = feature_extractor.transform(x_test)
    # shuffle test data because anomalies are all at the beginning
    random.seed(123)
    idx = np.arange(x_test.shape[0])
    random.shuffle(idx)
    x_test = x_test[idx, :]
    y_test = y_test[idx]
    return x_train, y_train, x_test, y_test


def record_result(rst_all, item, item_value, new):
    if item not in RST_HEADER:
        print(f'Wrong item({item}), valid item: {RST_HEADER}')
        return
    idx = np.where(RST_HEADER == item)[0][0]
    if new:
        rst_one = ['-' for i in range(len(RST_HEADER))]
        rst_one[idx] = item_value
        rst_all.append(rst_one)
    else:
        rst_all[len(rst_all) - 1][idx] = item_value
    return rst_all


def record_data_info(rst_all, data_name, feature_extract, feature_type, x_train, y_train, x_test, y_test, new=True):
    rst_all = record_result(rst_all, 'data_name', data_name, new)
    rst_all = record_result(rst_all, 'feature_extract', feature_extract, new=False)
    rst_all = record_result(rst_all, 'feature_type', feature_type, new=False)
    rst_all = record_result(rst_all, 'n_train_obj', x_train.shape[0], new=False)
    rst_all = record_result(rst_all, 'n_train_nor', np.sum(y_train==0), new=False)
    rst_all = record_result(rst_all, 'n_train_ano', np.sum(y_train==1), new=False)
    rst_all = record_result(rst_all, 'n_train_feature', x_train.shape[1], new=False)
    rst_all = record_result(rst_all, 'n_test_obj', x_test.shape[0], new=False)
    rst_all = record_result(rst_all, 'n_test_nor', np.sum(y_test == 0), new=False)
    rst_all = record_result(rst_all, 'n_test_ano', np.sum(y_test == 1), new=False)
    rst_all = record_result(rst_all, 'n_test_feature', x_test.shape[1], new=False)
    return rst_all


def record_evaluation(rst_all, method, dep_rel, dep_scoring, n_feature, y_true, score=None, n_report=None, y_pred=None,
                      verbose=1, new=False):
    TP, FP, FN, roc_auc, pr_auc, precision, recall, f1 = evaluate(y_true, score, n_report, y_pred, verbose)
    rst_all = record_result(rst_all, 'method', method, new=False)
    rst_all = record_result(rst_all, 'dep_rel', dep_rel, new=False)
    rst_all = record_result(rst_all, 'dep_scoring', dep_scoring, new=False)
    rst_all = record_result(rst_all, 'n_feature', n_feature, new=False)
    rst_all = record_result(rst_all, 'n_report', n_report, new=False)
    rst_all = record_result(rst_all, 'TP', TP, new=False)
    rst_all = record_result(rst_all, 'FP', FP, new=False)
    rst_all = record_result(rst_all, 'FN', FN, new=False)
    rst_all = record_result(rst_all, 'roc_auc', roc_auc, new=False)
    rst_all = record_result(rst_all, 'pr_auc', pr_auc, new=False)
    rst_all = record_result(rst_all, 'precision', precision, new=False)
    rst_all = record_result(rst_all, 'recall', recall, new=False)
    rst_all = record_result(rst_all, 'f1', f1, new=False)
    return rst_all


def save_result(filename, rst_all, header, append=True):
    if os.path.exists(filename):
        if not append:
            f = open(filename, 'wb')
            header_need = True
        else:
            f = open(filename, 'ab')
            header_need = False

    else:
        f = open(filename, 'xb')
        header_need = True

    if header_need:
        header = ','.join(header)
        np.savetxt(f, rst_all, delimiter=',', fmt='%s', header=header)
    else:
        np.savetxt(f, rst_all, delimiter=',', fmt='%s')
    f.close()
    return

