from sklearn.metrics import make_scorer, accuracy_score
import numpy as np


def tresh_gacc_scorer(cutoff):
    return make_scorer(tresh_gacc, cutoff=cutoff).set_score_request(groups=True)


def tresh_gacc(y_true, y_pred, *, groups=None, cutoff=0.25):
    yg_true = group_predict(y_true, groups=groups, cutoff=cutoff)
    yg_predict = group_predict(y_pred, groups=groups, cutoff=cutoff)

    return accuracy_score(yg_true, yg_predict)


def group_predict(y_pred, groups=None, cutoff=0.25, pos_label=1, neg_label=0):
    groups = np.array(groups)
    y_pred = np.array(y_pred)
    identifier = np.array(list(set(groups)))
    result = []
    for i in identifier:
        sel = groups == i
        positive = np.where(y_pred[sel] == pos_label, 1, 0)
        if np.mean(positive) >= cutoff:
            result.append(pos_label)
        else:
            result.append(neg_label)
    return np.array(result)
