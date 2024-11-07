import numpy as np


def encode_rle(x):
    mask = np.array(np.not_equal(x, np.roll(x, 1)))
    mask[0] = True
    elems = x[mask]
    indexes = np.where(mask)[0]
    if (len(indexes) == 0):
        return np.array([x[0]]), np.array([len(x)])
    lens = np.diff(indexes)
    lens = np.append(lens, len(x) - indexes[-1])
    return elems, lens


class RleSequence:

    def __init__(self, input_sequence):
        self._elems, self._lens = encode_rle(input_sequence)

    def __iter__(self):
        els = self._elems
        lens = self._lens
        while len(lens) > 0:
            yield els[0]
            lens[0] -= 1
            if lens[0] == 0:
                lens = lens[1:]
                els = els[1:]

    def __getitem__(self, k):
        if isinstance(k, slice):
            return np.repeat(self._elems, self._lens)[k]
        else:
            cum = np.cumsum(self._lens)
            while k < 0:
                k += cum[-1]
            i = np.where(cum > k)[0][0]
            return int(self._elems[i])

    def __contains__(self, s):
        return s in self._elems
