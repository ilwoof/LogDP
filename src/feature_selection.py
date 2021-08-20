
import numpy as np
import re
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from joblib import dump, load
import sys

sys.path.append('../')
import os
from utils import load_data, save_result
from utils import DATA_PATH
import random
# =========== setting ========
random.seed(123)
TRAIN = False
SAMPLE_TRN = True
feature_extract = 'trn'
feature_type = 'raw' # 'wn'

# transform the MB format
def format_mb(mb_one):
    # mb_one = '1  2 3 4   '
    mb_t = mb_one.strip()  # strip remove leading and tailing whitespace
    mb_t = mb_t.replace("  ", " ")  # replace two whitespace to one whitespace
    mb_t = mb_t.split(" ")
    mb_t = [int(i) for i in mb_t]
    return mb_t


def read_mb(filename):
    filename = open(filename, "r")
    mbs_raw = filename.read()
    filename.close()

    # compute how many lines in mbs_raw
    # format in each line: tgt_variable  mb1 mb2 ... \n
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


if __name__ == '__main__':

    for data_name in ['HDFS', 'bgl_1hourlogs', 'bgl_100logs', 'bgl_20logs',
                      'spirit_1hourlogs', 'spirit_100logs', 'spirit_20logs']: 

        output_dir = f"{DATA_PATH}/{data_name}/{feature_type}"
        if not os.path.exists(f"{DATA_PATH}/{data_name}"):
            os.mkdir(f"{DATA_PATH}/{data_name}")
            os.mkdir(f"{DATA_PATH}/{data_name}/{feature_type}")

        rst_all = []
        print(
            f'\n========== data={data_name}  feature_extract={feature_extract} feature_type={feature_type} ============')
        x_train, y_train, x_test, y_test = load_data(data_name, feature_extract, feature_type)

        x_train = x_train[y_train == 0, :]
        y_train = y_train[y_train == 0]

        all0_idx = np.load(os.path.join(DATA_PATH, f'{data_name}_E{feature_extract}_all0_idx.npy'))
        dep_idx = np.delete(np.arange(x_train.shape[1]), all0_idx)

        print(f"{data_name} = {x_train.shape[0]}")

        if SAMPLE_TRN:
            if x_train.shape[0] <= 500:
                n_sample_trn = x_train.shape[0]
            elif int(x_train.shape[0] * 0.05) < 500:
                n_sample_trn = 500
            else:
                n_sample_trn = int(x_train.shape[0] * 0.05)
                
            trn_smp_idx = np.random.choice(np.arange(x_train.shape[0]), n_sample_trn, replace=False)
            x_train = x_train[trn_smp_idx, :]
            y_train = y_train[trn_smp_idx]

        x_train = x_train[:, dep_idx]
        print(f"x_train = {x_train.shape}")
        y_train = np.array(y_train).flatten()

        print(f"In the train set, total data num = {len(y_train)}")
        print(f"In the train set, anomaly num = {np.count_nonzero(y_train)}")

        # save samples to txt file for MB learning
        mb_file_name = '_training_data_mb.txt'

        if not os.path.exists(f'{output_dir}/{data_name}_training_label.txt'):
            np.savetxt(f'{output_dir}/{data_name}_training_label.txt', y_train, delimiter=" ")

        if not os.path.exists(f'{output_dir}/{data_name}_training_data.txt'):
            np.savetxt(f'{output_dir}/{data_name}_training_data.txt', x_train, delimiter=" ")
        else:
            x_train = np.loadtxt(f'{output_dir}/{data_name}_training_data.txt', delimiter=" ")
            y_train = np.loadtxt(f'{output_dir}/{data_name}_training_label.txt', delimiter=" ")
