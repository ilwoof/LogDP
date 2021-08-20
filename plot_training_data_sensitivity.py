import numpy as np
import pandas as pd
from matplotlib import pyplot

if __name__ == '__main__':

    Data_Path = '..'
    df = pd.read_csv(f'{Data_Path}/training_data_sensitivity.csv', delimiter=',')
    HDFS = df[df['data'].str.contains('HDFS')]
    BGL_1hourlogs = df[df['data'].str.contains('bgl_1hourlogs')]
    BGL_100logs = df[df['data'].str.contains('bgl_100logs')]
    BGL_20logs = df[df['data'].str.contains('bgl_20logs')]

    log_datasets = {'HDFS': HDFS, 'bgl_1hourlogs': BGL_1hourlogs, 'bgl_100logs': BGL_100logs, 'bgl_20logs': BGL_20logs}
    # axis labels
    fig, ax = pyplot.subplots(figsize=(5, 3))
    pyplot.xlabel('The Size of Training Data')
    pyplot.xticks(rotation=45)
    pyplot.ylabel('ROC-AUC Score')
    # show the legend
    pyplot.ylim(0, 1.1, 0.2)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    for data_name, df in log_datasets.items():
        df["data"] = df["data"].str.replace(f"{data_name}_", "")
        pyplot.plot(df['data'], df['roc_auc'], marker='.', label=f'{data_name}')
        pyplot.legend()
    pyplot.savefig(f"./training_data_sensitivity.png", bbox_inches="tight")
