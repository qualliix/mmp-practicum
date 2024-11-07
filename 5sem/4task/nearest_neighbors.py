import sklearn.neighbors as sn
import numpy as np
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    np.int = int

    def __init__(self, k=10, strategy="my_own", metric="euclidean", weights=False, test_block_size=10):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.model = None
        if strategy != "my_own":
            self.model = sn.NearestNeighbors(
                n_neighbors=self.k, algorithm=strategy, metric=metric)

    def fit(self, X, y):
        if self.strategy != "my_own":
            self.model = self.model.fit(X)
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def find_kneighbors(self, X, return_distance=True):
        if self.model:
            return self.model.kneighbors(X, self.k, return_distance=return_distance)
        if self.metric == "cosine":
            distance = cosine_distance
        else:
            distance = euclidean_distance
        if self.test_block_size >= X.shape[0]:
            D = distance(X, self.X_train)
            indexes = np.argsort(D, axis=1)[:, :self.k]
            if return_distance:
                distances = np.sort(D, axis=1)[:, :self.k]
                return distances, indexes
            return indexes
        indexes = np.empty((X.shape[0], self.k))
        length = X.shape[0] // self.test_block_size
        if return_distance:
            distances = np.empty((X.shape[0], self.k))
            for i in range(length):
                D = distance(X[i*self.test_block_size: (i+1) *
                             self.test_block_size], self.X_train)
                indexes[i*self.test_block_size: (i+1) * self.test_block_size] = np.argsort(D)[
                    :, :self.k]
                distances[i*self.test_block_size: (i+1) *
                          self.test_block_size] = np.sort(D)[:, :self.k]
                del D
            D = distance(X[length*self.test_block_size:], self.X_train)
            indexes[length*self.test_block_size:] = np.argsort(D)[:, :self.k]
            distances[length*self.test_block_size:] = np.sort(D)[:, :self.k]
            return distances, indexes
        for i in range(length):
            indexes[i*self.test_block_size: (i+1) *
                    self.test_block_size] = np.argsort(distance(X[i*self.test_block_size: (i+1) *
                                                                  self.test_block_size], self.X_train))[:, :self.k]
        indexes[length*self.test_block_size:] = np.argsort(
            distance(X[length*self.test_block_size:], self.X_train))[:, :self.k]
        return indexes

    def predict(self, X):
        if self.weights:
            distances, indexes = self.find_kneighbors(X)
            weights = 1/(distances + 10**(-5))
            cl = np.vectorize(lambda x: self.y_train[int(x)])
            classes = cl(indexes)
            predictions = np.empty((len(self.classes), X.shape[0]))
            for i, c in enumerate(self.classes):
                predictions[i] = np.sum(
                    np.where(classes == c, weights, 0), axis=1)
        else:
            indexes = self.find_kneighbors(X, return_distance=False)
            cl = np.vectorize(lambda x: self.y_train[int(x)])
            classes = cl(indexes)
            predictions = np.empty((len(self.classes), X.shape[0]))
            for i, c in enumerate(self.classes):
                predictions[i] = np.sum(np.where(classes == c, 1, 0), axis=1)
        return self.classes[np.argmax(predictions, axis=0)]
