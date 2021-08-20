import os
if os.path.basename(os.getcwd()) == 'LogDependency':
    os.chdir(os.getcwd() + '/src')
print(f'current working directory: {os.getcwd()}')
import numpy as np
import pandas as pd
import random
from utils import DATA_PATH, RST_HEADER
from utils import load_data, record_data_info, record_evaluation, save_result
from lopad import lopad
from timeit import default_timer as timer
import argparse
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

random.seed(123)
SAVE_RST = True
feature_extract = 'trn'  # trn, tst, all
dep_rel = 'mb'  # all, mb, cor...
cor_thd = 0.5
feature_type = 'wn' # raw: raw count;  'w': weighted;  'n': normalized;  'wn': weighted+normalized
dep_scoring = 'ps' #'cityblock' #cityblock, ps
SAMPLE_TRN = True

parser = argparse.ArgumentParser(description='Detect the anomalies in system logs through dependency relationship.')
parser.add_argument('--algorithm', default=None, type=str, help='The name of the model.', required=True)
parser.add_argument('--sample', default=0.05, type=float, help='The number of the training data.')
paras = parser.parse_args()
algorithm_model = paras.algorithm
sample_trn = paras.sample

method_base = ''
if SAMPLE_TRN == True:
    method_base = method_base + '_sample'

for data_name in ['HDFS']:
                 # 'bgl_1hourlogs', 'bgl_100logs', 'bgl_20logs', 'spirit_1hourlogs', 'spirit_100logs', 'spirit_20logs']:

    all_start = timer()
    rst_all = []
    print(f'\n========== data={data_name}/feature_extract={feature_extract}/feature_type={feature_type} ============')
    x_train, y_train, x_test, y_test = load_data(data_name, feature_extract, 'raw')
    rst_all = record_data_info(rst_all, data_name, feature_extract, 'raw',
                               x_train, y_train, x_test, y_test, new=True)
    x_train = x_train[y_train == 0, :]
    # if break all0 rules, label as anomalies
    all0_idx = np.load(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_all0_idx.npy'))
    all0_score = np.sum(x_test[:, all0_idx], axis=1)
    all0_detect_idx = np.where(all0_score > 0)[0]
    y_pred = np.zeros(len(y_test))
    y_pred[all0_detect_idx] = 1

    # check dependency for the rest feature
    if feature_type != 'raw':
        x_train, y_train, x_test, y_test = load_data(data_name, feature_extract, feature_type)
        x_train = x_train[y_train == 0, :]
     
    y_train = y_train[y_train == 0]

    if SAMPLE_TRN:
        # if x_train.shape[0] <= 500:
        #     n_sample_trn = x_train.shape[0]
        # elif int(x_train.shape[0] * 0.05) < 500:
        #     n_sample_trn = 500
        # else:
        #     n_sample_trn = int(x_train.shape[0] * 0.05)

        if sample_trn < 1.0:
            n_trn = int(x_train.shape[0] * sample_trn)
            if n_trn < 100:
                n_trn = 100
        else:
            if int(sample_trn) < 100:
                n_trn = 100
            elif x_train.shape[0] < int(sample_trn):
                n_trn = x_train.shape[0]
        trn_smp_idx = np.random.choice(np.arange(x_train.shape[0]), n_trn, replace=False)
        x_train = x_train[trn_smp_idx, :]

    dep_idx = np.delete(np.arange(x_test.shape[1]), all0_idx)

    dep_score, training_time, testing_time = lopad(data_name, x_train[:, dep_idx], x_test[:, dep_idx], y_test,
                                                   dep_model='mlp', scoring=dep_scoring, dep_rel=dep_rel,
                                                   program_start_time=all_start)
    print(f'----- feature: all0#={len(all0_idx)}/dependency={len(dep_idx)}/all#={x_test.shape[1]})-----')
    # final results
    final_score = np.copy(dep_score)
    final_score[all0_detect_idx] = np.max(dep_score)+100

    #np.savetxt(f'./curve_data/tst_score_{data_name}_LogDependency.txt', final_score, delimiter=',')
    #np.savetxt(f'./curve_data/tst_y_true_{data_name}_logDependency.txt', y_test, delimiter=',')
    prs, res, thresholds = precision_recall_curve(y_test, final_score)
    # calculate precision-recall AUC

    print(f"roc_score = {round(roc_auc_score(y_test, final_score),3)}, pr_auc ={round(auc(res, prs),3)}")
    time_evaluation = open(f"./running_time_records.txt", 'a')
    print(f"{data_name}_{sample_trn},{algorithm_model},{training_time},{testing_time}", file=time_evaluation)

    if SAVE_RST:
        result_file_name = f'logdependency_precision_recall_in_topk.csv'
        title_str = np.array(['data-sample', 'recall-1%', '3%', '5%', '7%', '9%', '11%',
                              'precision-1%', '3%', '5%', '7%', '9%', '11%'])
        vv_sorted = np.argsort(-final_score)
        topk_recall_list = []
        topk_precision_list = []
        for i in np.arange(0.01, 0.13, 0.02):
            top = int(len(vv_sorted) * i)
            tp = np.sum(y_test[vv_sorted[:top]])
            recall = round(tp * 100 / np.sum(y_test), 3)
            precision = round(tp * 100 / top, 3)
            topk_recall_list.append(f"{recall}")
            topk_precision_list.append(f"{precision}")

        result_list = [f'{data_name}_{sample_trn},' + ','.join(topk_recall_list) + ',' + ','.join(topk_precision_list)]
        save_result(result_file_name, result_list, title_str)
