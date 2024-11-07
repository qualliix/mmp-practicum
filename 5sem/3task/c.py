import numpy as np


def replace(col):
    isNan = np.isnan(col)
    mean = 0
    if (not np.all(isNan)):
        mean = np.mean(col, where=np.logical_not(isNan))
    return np.nan_to_num(col, nan=mean)


def replace_nan_to_means(X):
    return np.apply_along_axis(replace, 0, X)
