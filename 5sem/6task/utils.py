import numpy as np
import oracles

def grad_finite_diff(X, y, w, eps=1e-8, l2_coef=0):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    bl = oracles.BinaryLogistic(l2_coef=l2_coef)
    e = np.zeros(w.shape[0])
    e[0] = 1
    result = np.arange(w.shape[0])
    grad = np.vectorize(lambda r: (bl.func(X, y, w + eps * np.roll(e, r)) - bl.func(X, y, w)) / eps)
    return grad(result)