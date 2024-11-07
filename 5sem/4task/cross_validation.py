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


def predict(y_train, X_test, k, neighbors, weights):
    clas = np.unique(y_train)
    if weights:
        distances = neighbors[0][:, :k]
        indexes = neighbors[1][:, :k]
        weights = 1/(distances + 10**(-5))
        cl = np.vectorize(lambda x: y_train[int(x)])
        classes = cl(indexes)
        predictions = np.empty((len(clas), X_test.shape[0]))
        for i, c in enumerate(clas):
            predictions[i] = np.sum(
                np.where(classes == c, weights, 0), axis=1)
    else:
        indexes = neighbors[:, :k]
        cl = np.vectorize(lambda x: y_train[int(x)])
        classes = cl(indexes)
        predictions = np.empty((len(clas), X_test.shape[0]))
        for i, c in enumerate(clas):
            predictions[i] = np.sum(np.where(classes == c, 1, 0), axis=1)
    return clas[np.argmax(predictions, axis=0)]


def knn_cross_val_score(X, y, k_list, score=None, cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)
    res = dict()
    if "weights" in kwargs:
        weights = kwargs["weights"]
    else:
        weights = False
    for k in k_list:
        res[k] = []
    knn = KNNClassifier(k=k_list[-1], **kwargs)
    for train, test in cv:
        knn.fit(X[train], y)
        neighbors = knn.find_kneighbors(X[test], return_distance=weights)
        for k in k_list:
            pred = predict(y[train], X[test], k,
                           neighbors=neighbors, weights=weights)
            res[k].append(accuracy(y[test], pred))
    return res
