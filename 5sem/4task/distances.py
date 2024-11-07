import numpy as np


def cosine_distance(m1, m2):
    dot_product = np.dot(m1, m2.transpose())
    norm_m1 = np.linalg.norm(m1, ord=2, axis=1)
    norm_m2 = np.linalg.norm(m2, ord=2, axis=1)
    return 1 - dot_product / norm_m1[:, np.newaxis] / norm_m2


def euclidean_distance(m1, m2):
    return np.sqrt(-2*(np.dot(m1, m2.transpose()))
                   + np.diag(np.dot(m1, m1.transpose()))[:, np.newaxis]
                   + np.diag(np.dot(m2, m2.transpose())))
