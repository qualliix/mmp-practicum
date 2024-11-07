import numpy as np


def get_nonzero_diag_product(X):
    diag = np.diag(X)
    isNone = np.where(diag != 0)
    diag = np.where(diag != 0, diag, 1)
    if isNone[0].size == 0:
        return None
    return float(np.prod(diag))
