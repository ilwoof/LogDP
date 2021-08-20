import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, precision_recall_fscore_support
from matplotlib import pyplot

if __name__ == '__main__':

    Data_Path = './curve_data'
    for curve_type in ['roc_curve']:#['roc_curve', 'pre_curve']:
        for data_name in ['HDFS_session', 'bgl_1hourlogs', 'bgl_100logs', 'bgl_20logs',
                          'spirit_1hourlogs', 'spirit_100logs', 'spirit_20logs']:
            for approach in ['OCSVM', 'DL']: #['PCA', 'Invariant_Mining', 'OCSVM', 'LogCluster', 'DL','ADR', 'LogDependency']:
                if approach in ['OCSVM', 'DL']:
                    df = pd.read_csv(f'{Data_Path}/{approach}/tst_score_{data_name}_{approach}.csv')
                    pred_score = df['y_predict_score'].tolist()
                    y_test = df['y'].tolist()
                    y_pred = df['y_pred'].tolist()
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                    print(f"{approach}-{data_name}: precision = {round(precision,3)}, recall = {round(recall,3)}, f1 = {round(f1,3)}")
                    continue
                else:
                    pred_score = np.loadtxt(f'{Data_Path}/tst_score_{data_name}_{approach}.txt', delimiter=',')
                    y_test = np.loadtxt(f'{Data_Path}/tst_y_true_{data_name}_{approach}.txt', delimiter=',')

                if curve_type == 'pre_curve':
                    precision, recall, _ = precision_recall_curve(y_test, pred_score)
                    # plot the precision-recall curves
                    pyplot.plot(recall, precision, label=f'{approach}')
                    # axis labels
                    pyplot.xlabel('Recall')
                    pyplot.ylabel('Precision')
                    # show the legend
                    pyplot.legend(loc=4)
                else:
                    fpr, tpr, _ = roc_curve(y_test, pred_score)
                    # plot the roc curve for the model
                    pyplot.plot(fpr, tpr, label=f'{approach}')
                    # axis labels
                    pyplot.xlabel('False Positive Rate')
                    pyplot.ylabel('True Positive Rate')
                    # show the legend
                    pyplot.legend(loc=4)
                    # show the plot
            pyplot.savefig(f"./{data_name}_{curve_type}.png")
            pyplot.close(fig='all')
