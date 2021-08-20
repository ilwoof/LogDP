import os
if os.path.basename(os.getcwd()) == 'loglizer':
    os.chdir(os.getcwd() + '\\src')
print(f'current working directory: {os.getcwd()}')

import numpy as np
import random
# import pickle
# from loglizer import dataloader, preprocessing
from utils import DATA_PATH
from utils import load_data

# <editor-fold desc="==== setting ====">
random.seed(123)
# FEATURE_EXTRACT = 'trn'   #'trn'  'tst'  'all'
# </editor-fold>

# <editor-fold desc="==== load data ====">

for feature_extract in ['trn', 'tst', 'all']:

    for data_name in ['HDFS', 'bgl_1hourlogs', 'bgl_20logs', 'bgl_100logs',
                      'spirit_1hourlogs', 'spirit_20logs', 'spirit_100logs']:

        print(f'\n------ data={data_name} extract_feature={feature_extract} ------')
        x_train, y_train, x_test, y_test = load_data(data_name, feature_extract)

        # check duplicated columns and all 0 columns
        x_train_nor = x_train[y_train == 0, :]
        dup_ind = np.ones(x_train_nor.shape[1]) * -1
        all0_idx = []
        for i in range(x_train_nor.shape[1]):
            # check all 0
            if np.all(x_train_nor[:, i] == 0):
                all0_idx.append(i)
                continue
            # check duplicate
            for j in range(i+1, x_train_nor.shape[1]):
                if dup_ind[j] != -1:
                    continue
                if np.all(x_train_nor[:, i] == x_train_nor[:, j]) and i != j:
                    # print(f'{j} same as {i}: {np.unique(x_train_nor[:,j])}')
                    dup_ind[j] = i
        all0_idx = np.array(all0_idx)
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_all0_idx'), all0_idx)
        dup_ind = dup_ind.astype(np.int32)
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_dup_ind'), dup_ind)

        # check anomalies w.r.t all 0 and duplicated colomns
        ano_dup_idx = []
        nor_dup_idx = []
        ano_all0_idx = []
        nor_all0_idx = []
        for i in range(x_test.shape[1]):
            # i = 3
            # check all 0
            if i in all0_idx:
                if np.any(x_test[:, i]):
                    # print(f'found all0 anoamly: {i}')
                    not0_idx = np.where(x_test[:, i])[0]
                    idx = np.where(y_test[not0_idx] == 1)
                    if len(idx) > 0:
                        ano_all0_idx = ano_all0_idx + list(not0_idx[idx])
                    idx = np.where(y_test[not0_idx] == 0)
                    if len(idx) > 0:
                        nor_all0_idx = nor_all0_idx + list(not0_idx[idx])
                continue

            # check duplicate column
            if dup_ind[i] == -1: continue
            dup_from = dup_ind[i]
            dup_to = i
            diff_idx = np.where( x_test[:, dup_to] != x_test[:, dup_from] )[0]
            if len(diff_idx) == 0: continue
            idx = np.where(y_test[diff_idx] == 1)
            if len(idx) > 0:
                ano_dup_idx = ano_dup_idx + list(diff_idx[idx])
            idx = np.where(y_test[diff_idx] == 0)
            if len(idx) > 0:
                nor_dup_idx = nor_dup_idx + list(diff_idx[idx])

        ano_all0_idx = np.unique(np.array(ano_all0_idx))
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_ano_all0_idx'), ano_all0_idx)
        nor_all0_idx = np.unique(np.array(nor_all0_idx))
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_nor_all0_idx'), nor_all0_idx)
        ano_dup_idx = np.unique(np.array(ano_dup_idx))
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_ano_dup_idx'), ano_dup_idx)
        nor_dup_idx = np.unique(np.array(nor_dup_idx))
        np.save(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_nor_dup_idx'), nor_dup_idx)


        print(f'train_data={x_train_nor.shape[0]}*{x_train_nor.shape[1]}    '
              f'#all0_set={len(all0_idx)}    #dup_set={np.sum(dup_ind != -1)}    ')
        print(f'test_data={x_test.shape[0]}*{x_test.shape[1]}    #ano_test={np.sum(y_test == 1)}    '
              f'#ano_all0/#nor_all0={len(ano_all0_idx)}/{len(nor_all0_idx)}     '
              f'#ano_dup/#nor_dup={len(ano_dup_idx)}/{len(nor_dup_idx)}')

