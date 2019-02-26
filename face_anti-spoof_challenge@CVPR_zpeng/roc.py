import numpy as np

from scipy import interpolate

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


#reference from https://github.com/JinghuiZhou/awesome_face_antispoofing/edit/master/roc.py
def cal_metric(groundTruth, predicted):
	fpr, tpr, thresholds = roc_curve(groundTruth, predicted)
	y = (tpr)
	x = (fpr)
	z = tpr +fpr
	tpr = tpr.reshape((tpr.shape[0],1))
	fpr = fpr.reshape((fpr.shape[0],1))
	xnew = np.arange(0, 1, 0.00000001)
	func = interpolate.interp1d(x, y)

	ynew = func(xnew)

	znew = abs(xnew + ynew-1)

	eer=xnew[np.argmin(znew)]

#	print('EER',eer)

	FPR = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}

	TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
	for i, (key, value) in enumerate(FPR.items()):

		index = np.argwhere(xnew == value)

		score = ynew[index] 

		TPRs[key] = float(np.squeeze(score))
#	    print(key, score)
	if 0:
		plt.plot(xnew, ynew)

		plt.show()
	auc = roc_auc_score(groundTruth, predicted)
	return eer,TPRs, auc,{'x':xnew, 'y':ynew}
