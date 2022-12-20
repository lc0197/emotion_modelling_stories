from typing import Dict

from scipy.stats import pearsonr

from sklearn.metrics import *
import numpy as np


def metric_arr_to_dict(metric_arr, idx2label) -> Dict[str, float]:
    '''
    Converts an array containing a metric value for different labels into a dict {l1:v1,...}
    @param name: name of the metric
    @param metric_arr: the array to convert
    @param idx2label: mapping from array position to label name
    @return: dict with labels as keys
    '''
    assert metric_arr.ndim == 1
    return {f'{idx2label[i]}': metric_arr[i] for i in range(metric_arr.shape[0])}


def ccc(preds, labels):
    """
    Concordance Correlation Coefficient
    @param preds: 1D np array
    @param labels: 1D np array
    @return CCC
    """

    preds_mean, labels_mean = np.mean(preds), np.mean(labels)
    cov_mat = np.cov(preds, labels)
    covariance = cov_mat[0, 1]
    preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

    ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)
    return ccc


def tales_metrics_continuous(predictions, gold_standards, labels) -> Dict[str, float]:
    '''

    Calculates CCC, Pearson correlation and MSE
    @param predictions:
    @param gold_standards:
    @param labels:
    @return:
    '''
    dct = {}
    for i, l in enumerate(labels):
        l_dict = {}
        preds = predictions[:, i]
        gss = gold_standards[:, i]
        l_dict['CCC'] = ccc(preds, gss)
        l_dict['Pearson'] = pearsonr(preds, gss)[0]
        l_dict['MSE'] = mean_squared_error(gss, preds)
        dct[l] = l_dict
    return dct
