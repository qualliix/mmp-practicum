import numpy as np


class BatchGenerator:
    def __init__(self, list_of_sequences, batch_size, shuffle=False):
        self._list = list_of_sequences
        self._batch_size = batch_size
        if shuffle:
            for l in self._list:
                np.random.shuffle(l)

    def __iter__(self):
        for i in range(0, len(self._list[0]), self._batch_size):
            res = []
            for j in range(len(self._list)):
                res.append(self._list[j][i:i+self._batch_size])
            yield res
