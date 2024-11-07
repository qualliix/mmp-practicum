import numpy as np
import scipy as sc


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return (np.sum(np.logaddexp(0, -y * (X @ w)))/X.shape[0]
                + (self.l2_coef/2) * np.dot(w, w))
        return super().func(w)

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        vec = (X @ w) * y
        dot = np.dot((-y)*sc.special.expit(vec) * np.exp(-vec), X)
        return (dot/ X.shape[0]
                + self.l2_coef * w)
        return super().grad(w)
np.random.seed(1693)
l2_coef = np.random.randint(0, 10)
l, d = 1000, 10
my_oracle = BinaryLogistic(l2_coef=l2_coef)
X = np.random.random((l, d))
y = np.random.randint(0, 2, l) * 2 - 1
w = np.random.random(d)
print(X.shape, y.shape, w.shape)
res_grad = my_oracle.grad(X, y, w)
res = " ".join([str(x) for x in res_grad])
print(res)