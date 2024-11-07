import numpy as np


def calc_expectations(h, w, X, Q):
    # По строкам
    cumsum = np.cumsum(Q, axis=1)
    sub = np.roll(cumsum, w, axis=1)
    slice_ = np.s_[:, w:]
    mask = np.ones_like(sub, dtype=bool)
    mask[slice_] = False
    sub[mask] = 0
    cumsum = cumsum - sub
    # по столбцам
    cumsum = np.cumsum(cumsum, axis=0)
    sub = np.roll(cumsum, h, axis=0)
    slice_ = np.s_[h:]
    mask = np.ones_like(sub, dtype=bool)
    mask[slice_] = False
    sub[mask] = 0
    cumsum = cumsum - sub
    return X*cumsum
