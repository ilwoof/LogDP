
import numpy as np
import os
import re
from utils import save_result

feature_type = 'raw'

def read_tm_file(filename):
    filename = open(filename, "r")
    lines = filename.readlines()
    filename.close()

    time_list = []

    for line in lines:
        if line != ' ' and line != '\n':  # Remove space and line-change characters
            element_time = line.split(':')[1]
            element_time = element_time.split('\n')[0]
            time_list.append(float(element_time))

    return np.sum(np.array(time_list))

if __name__ == '__main__':

    total_time = []
    # data_name = ['HDFS', 'bgl_1hourlogs', 'bgl_100logs', 'bgl_20logs',
    #              'spirit_1hourlogs', 'spirit_100logs', 'spirit_20logs']
    data_name = ['spirit_20logs']
    for project in data_name:
        for i in range(5):
            data_path = f'../data/{project}/{feature_type}/'
            file_name = f"{data_path}/{project}_training_data_{i}_tm.txt"
            if os.path.exists(file_name):
                split_time = read_tm_file(file_name)
                total_time.append(split_time)
            elif i == 0:
                file_name = f"{data_path}/{data_name}_training_data_tm.txt"
                if os.path.exists(file_name):
                    split_time = read_tm_file(file_name)
            else:
                print(f"{file_name} does not exist!")

        result_file_name = f'LogDP_mb_training_time.csv'
        title_str = np.array(['model', 'MB_time'])
        result_list = [f'{project},{round(np.sum(np.array(total_time)),3)}']
        save_result(result_file_name, result_list, title_str)
        print(round(np.sum(np.array(total_time)),3))
