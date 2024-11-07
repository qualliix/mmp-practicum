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
