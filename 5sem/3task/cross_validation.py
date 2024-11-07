import numpy as np
from nearest_neighbors import KNNClassifier
np.int = int


def kfold(n, n_folds):
    nums = np.arange(n)
    test = np.array_split(nums, n_folds)
    res = []
    for t in test:
        train = [x for x in nums if x not in t]
        res.append((np.array(train), np.array(t)))
    return res


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def knn_cross_val_score(X, y, k_list, score='accuracy', cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 10)
    res = dict()
    for k in k_list:
        scores = []
        for train, test in cv:
            knn = KNNClassifier(k=k, **kwargs)
            knn.fit(X[train], y[train])
            scores.append(accuracy(y[test], knn.predict(X[test])))
        res[k] = scores
    return res
