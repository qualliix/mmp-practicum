import numpy as np


def get_max_before_zero(x):
    zeros = np.where(x == 0)[0]
    if (len(zeros) == 0) or (len(x) == 1) and (len(zeros) == 1):
        return None
    if (zeros[-1] == len(x)-1):
        zeros = np.delete(zeros, -1)
    return int(np.max(x[(zeros + 1).astype(int)]))
