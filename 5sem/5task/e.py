import numpy as np


class Polynomial:
    def __init__(self, *args):
        self._coefs = list(args)
        self._degree = len(self._coefs) - 1

    def __call__(self, x):
        return np.sum(np.cumprod(
            np.concatenate(([1], np.tile([x], self._degree)))) * self._coefs)

    def getCoefs(self):
        return self._coefs

    def setCoefs(self, new_coefs):
        if not (isinstance(new_coefs, tuple) or isinstance(new_coefs, list)):
            raise TypeError
        self._coefs = new_coefs
        self._degree = len(self._coefs) - 1

    def __setitem__(self, k, v):
        self._coefs[k] = v

    def __getitem__(self, k):
        return self._coefs[k]

    coefs = property(getCoefs, setCoefs)


class IntegerPolynomial(Polynomial):
    def __init__(self, *args):
        super().__init__(*args)
        self._coefs = list(np.round(self._coefs).astype(int))

    def setCoefs(self, new_coefs):
        if not (isinstance(new_coefs, tuple) or isinstance(new_coefs, list)):
            raise TypeError
        self._coefs = list(np.round(new_coefs).astype(int))
        self._degree = len(self._coefs) - 1

    def __setitem__(self, k, v):
        self._coefs[k] = np.round(v).astype(int)

    coefs = property(Polynomial.getCoefs, setCoefs)
