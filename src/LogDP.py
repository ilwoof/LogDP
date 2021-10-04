import os

if os.path.basename(os.getcwd()) == 'LogDP':
    os.chdir(os.getcwd() + '/src')
print(f'current working directory: {os.getcwd()}')
import numpy as np
import random
from utils import DATA_PATH
from utils import load_data
from lopad import lopad_get_exp

random.seed(123)

# the current MB is learned of setting: feature_extract = 'trn'
feature_extract = 'trn'  # trn tst all
beta = 1
feature_type = 'wn'

for data_name in ['HDFS']:
                  # ,'bgl_1hourlogs', 'bgl_100logs'
                  # ,'spirit_1hourlogs'
                  # ,'spirit_100logs', 'bgl_20logs', 'spirit_20logs']:
    print(f'\n=========== data={data_name}   feature_extract={feature_extract} ==============')
    x_trn_all, y_trn, x_tst, y_tst = load_data(data_name, feature_extract, feature_type=feature_type)
    x_trn_nor_all = x_trn_all[y_trn == 0, :]
    trn_num = int(x_trn_nor_all.shape[0] * 2 / 3)
    n_trn = x_trn_nor_all.shape[0] if x_trn_nor_all.shape[0] <= trn_num else int(x_trn_nor_all.shape[0] * 0.05)
    x_trn = x_trn_nor_all[:n_trn, :]
    x_val = x_trn_nor_all[n_trn:, :]

    ## --------- get expected values -----------
    trn_exp, tst_exp, val_exp = lopad_get_exp(data_name, x_trn, x_tst, x_val, dep_model='mlp')

    ## --------- detection ---------------
    val_dev = None if val_exp is None else np.abs(x_val - val_exp)
    trn_dev = np.abs(x_trn - trn_exp)
    tst_dev = np.abs(x_tst - tst_exp)
    thd_all = np.zeros(x_tst.shape[1])
    thd_all = np.max(val_dev, axis=0) * beta

    y_pred = np.zeros(x_tst.shape[0])
    for i in range(len(thd_all)):
        idx = np.where(tst_dev[:, i] > thd_all[i])[0]
        y_pred[idx] = 1

    # ------- evaluation -------
    precision = np.sum(np.logical_and(y_pred == 1, y_tst == 1)) / np.sum(y_pred == 1)
    recall = np.sum(np.logical_and(y_pred == 1, y_tst == 1)) / np.sum(y_tst == 1)
    f1 = 2 * precision * recall / (precision + recall)
    print(f'data_name={data_name}   beta={beta}   n_report={np.sum(y_pred == 1)}/{np.sum(y_tst == 1)}   '
          f'precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}')

    # np.save(os.path.join(DATA_PATH, f'{data_name}_{trn_num}_{feature_type}_trn_exp_f{round(f1, 2)}.npy'), trn_exp)
    # np.save(os.path.join(DATA_PATH, f'{data_name}_{trn_num}_{feature_type}_tst_exp_f{round(f1, 2)}.npy'), tst_exp)
    # np.save(os.path.join(DATA_PATH, f'{data_name}_{trn_num}_{feature_type}_val_exp_f{round(f1, 2)}.npy'), val_exp)
