from oracles import BinaryLogistic
import time
import numpy as np
import scipy as sc


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.bl = BinaryLogistic(**kwargs)

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self.w = w_0
        self.history = {"time": [0], "func": [self.get_objective(X, y)]}
        for k in range(self.max_iter):
            s = time.time()
            grad = self.get_gradient(X, y)
            eta = self.step_alpha / k**self.step_beta
            self.w = self.w - eta*grad
            f = self.get_objective(X, y)
            self.history["time"].append(time.time() - s)
            self.history["func"].append(f)
            if abs(self.history["func"][-2] - f) < self.tolerance:
                break
        if trace:
            return self.history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X @ self.w)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        prob = sc.special.expit(X @ self.w)
        return np.concatenate((prob, 1 - prob)).T

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.bl.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.bl.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size=10, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.bl = BinaryLogistic(**kwargs)
        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        self.w = w_0
        self.history = {"time": [0], "func": [self.get_objective(X, y)], "weights_diff": [
            0], "epoch_num": [0]}
        count = 0
        lenght = len(X)
        np.random.seed(self.random_seed)
        for k in range(self.max_iter):
            s = time.time()
            for i in range(0, lenght, self.batch_size):
                grad = self.get_gradient(
                    X[i:i+self.batch_size], y[i:i+self.batch_size])
                eta = self.step_alpha / k**self.step_beta
                prev_w = self.w
                self.w = self.w - eta*grad
                count += self.batch_size
                epoch_num = count / lenght
                if (epoch_num - self.history["epoch_num"][-1]) < log_freq:
                    f = self.get_objective(
                        X[i:i+self.batch_size], y[i:i+self.batch_size])
                    self.history["time"].append(time.time() - s)
                    self.history["func"].append(f)
                    self.history["epoch_num"].append(epoch_num)
                    self.history["weights_diff"].append(
                        (prev_w - self.w) @ (prev_w - self.w))
                    if abs(self.history["func"][-2] - f) < self.tolerance:
                        break
        if trace:
            return self.history
# np.random.seed(10)
# clf = SGDClassifier(loss_function='binary_logistic', step_alpha=1,
#     step_beta=0, tolerance=1e-4, batch_size=100,  max_iter=6, l2_coef=0.1)
# l, d = 1000, 10
# X = np.random.random((l, d))
# y = np.random.randint(0, 2, l) * 2 - 1
# w = np.random.random(d)
# history = clf.fit(X, y, w_0=np.zeros(d), trace=True)
# print(' '.join([str(x) for x in history['epoch_num']]))
