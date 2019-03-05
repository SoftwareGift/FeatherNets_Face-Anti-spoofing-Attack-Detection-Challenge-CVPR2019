import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def cal_metric(target, predicted,show = False):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    _tpr = (tpr)
    _fpr = (fpr)
    tpr = tpr.reshape((tpr.shape[0],1))
    fpr = fpr.reshape((fpr.shape[0],1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y -1)
    eer = scale[np.argmin(znew)]
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index] 
        TPRs[key] = float(np.squeeze(score))
    auc = roc_auc_score(target, predicted)
    if show:
        plt.plot(scale, y)
        plt.show()
    return eer,TPRs, auc,{'x':scale, 'y':y}
