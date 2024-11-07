class CooSparseMatrix:
    def __init__(self, ijx_list, shape):
        self.matrix = dict()
        self._shape = shape
        self.lines = set()
        if (not isinstance(shape, tuple)) or (len(shape) != 2):
            raise TypeError
        if (not isinstance(ijx_list, list)):
            raise TypeError
        if (not isinstance(shape[0], int)
            or (shape[0] < 0)
            or (not isinstance(shape[1], int))
                or (shape[1] < 0)):
            raise TypeError
        for elem in ijx_list:
            if (not isinstance(elem, tuple)
                or (len(elem) != 3)
                or (not isinstance(elem[0], int))
                or (not isinstance(elem[1], int))
                    or ((not isinstance(elem[2], int)) and (not isinstance(elem[2], float)))):
                raise TypeError
            if ((elem[0] >= shape[0])
                or (elem[1] >= shape[1])
                or (elem[1] < 0)
                    or (elem[0] < 0)):
                raise TypeError
            pos = elem[:-1]
            if pos in self.matrix:
                raise TypeError
            else:
                if (elem[2] == 0):
                    continue
                self.lines.add(elem[0])
                self.matrix[pos] = elem[2]

    def __getitem__(self, index):
        if isinstance(index, int):
            if ((index >= self._shape[0]) or (index < 0)):
                raise TypeError
            if (index not in self.lines):
                return CooSparseMatrix(ijx_list=[], shape=(1, self._shape[1]))
            out = []
            for i in range(self._shape[1]):
                if ((index, i) in self.matrix):
                    out.append((0, i, self.matrix[(index, i)]))
            return CooSparseMatrix(ijx_list=out, shape=(1, self._shape[1]))
        if (isinstance(index, tuple)
                and len(index) == 2):
            if ((index[0] >= self._shape[0])
                or (index[1] >= self._shape[1])
                or (index[0] < 0)
                or (index[1] < 0)
                or (not isinstance(index[0], int))
                    or (not isinstance(index[1], int))):
                raise TypeError
            if (index in self.matrix):
                return self.matrix[index]
            else:
                return 0
        else:
            raise TypeError

    def __setitem__(self, index, value):
        if ((not isinstance(index, tuple)) or len(index) != 2):
            raise TypeError
        if ((type(index[0]) is not int)
            or (type(index[1]) is not int)
            or ((type(value) is not int)
                and (type(value) is not float))):
            raise TypeError
        if ((index[0] >= self._shape[0])
            or (index[1] >= self._shape[1])
            or (index[0] < 0)
                or (index[1] < 0)):
            raise TypeError
        if (value == 0):
            if (index in self.matrix):
                del self.matrix[index]
        else:
            self.matrix[index] = value
            self.lines.add(index[0])

    def __add__(self, other):
        out_list = []
        if (not isinstance(other, CooSparseMatrix)) or (self._shape != other._shape):
            raise TypeError
        for index in self.matrix:
            i, j = index
            if index in other.matrix:
                out_list.append((i, j, self.matrix[index]+other.matrix[index]))
            else:
                out_list.append((i, j, self.matrix[index]))
        for index in other.matrix:
            i, j = index
            if (index not in self.matrix):
                out_list.append((i, j, other.matrix[index]))
        return CooSparseMatrix(out_list, self._shape)

    def __sub__(self, other):
        out_list = []
        if (not isinstance(other, CooSparseMatrix)) or (self._shape != other._shape):
            raise TypeError
        for index in self.matrix:
            i, j = index
            if index in other.matrix:
                out_list.append(
                    (i, j, self.matrix[index] - other.matrix[index]))
            else:
                out_list.append((i, j, self.matrix[index]))
        for index in other.matrix:
            i, j = index
            if index not in self.matrix:
                out_list.append((i, j, -other.matrix[index]))
        return CooSparseMatrix(out_list, self._shape)

    def __mul__(self, value):
        out_list = []
        if value == 0:
            return CooSparseMatrix(out_list, self._shape)
        for i, j in self.matrix:
            if (self.matrix[(i, j)] == 0):
                continue
            out_list.append((i, j, self.matrix[(i, j)]*value))
        return CooSparseMatrix(out_list, self._shape)

    def __rmul__(self, value):
        out_list = []
        if value == 0:
            return CooSparseMatrix(out_list, self._shape)
        for i, j in self.matrix:
            if (self.matrix[(i, j)] == 0):
                continue
            out_list.append((i, j, self.matrix[(i, j)]*value))
        return CooSparseMatrix(out_list, self._shape)

    def getShape(self):
        return self._shape

    def printMatrix(self):
        print("  ", end=" ")
        for j in range(self._shape[1]):
            print(j, end=' ')
        print("\n---------------")
        for i in range(self._shape[0]):
            print(i, end='| ')
            for j in range(self._shape[1]):
                print(self[i, j], end=" ")
            print()

    def setShape(self, new_shape):
        if ((not isinstance(new_shape, tuple))
            or (len(new_shape) != 2)
            or (not isinstance(new_shape[0], int))
                or (not isinstance(new_shape[1], int))):
            raise TypeError
        if self._shape == new_shape:
            return
        if ((self._shape[0] * self._shape[1]) != (new_shape[0] * new_shape[1])):
            raise TypeError
        new_list = []
        for line, col in self.matrix:
            pos = line * self._shape[1] + col
            new_list.append(
                (pos//new_shape[1], pos % new_shape[1], self.matrix[line, col]))
        self._shape = new_shape
        self.matrix.clear()
        for i, j, x in new_list:
            self.matrix[i, j] = x

    shape = property(getShape, setShape)

    def getT(self):
        new_list = []
        new_shape = (self._shape[1], self._shape[0])
        for line, col in self.matrix:
            new_list.append((col, line, self.matrix[line, col]))
        return CooSparseMatrix(new_list, new_shape)

    def setT(self, t):
        raise AttributeError

    T = property(getT, setT)
