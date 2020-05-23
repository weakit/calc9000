import sympy
# TODO: Dot, Nothing
# TODO: check others


class RuleException(Exception):
    pass


class ListException(Exception):
    pass


class UnequalLengthException(ListException):
    pass


class List(tuple):
    # TODO: Pretty and LaTeX printing
    def __add__(self, other):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise UnequalLengthException
            for i in range(len(self)):
                temp_list.append(self[i] + other[i])
            return List(temp_list)
        for i in range(len(self)):
            temp_list.append(self[i] + other)
        return List(temp_list)

    __iadd__ = __add__

    def __radd__(self, other):
        temp_list = []
        for i in range(len(self)):
            temp_list.append(other + self[i])
        return List(temp_list)

    def __sub__(self, other):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise UnequalLengthException
            for i in range(len(self)):
                temp_list.append(self[i] - other[i])
            return List(temp_list)
        for i in range(len(self)):
            temp_list.append(self[i] - other)
        return List(temp_list)

    __isub__ = __sub__

    def __rsub__(self, other):
        temp_list = []
        for i in range(len(self)):
            temp_list.append(other - self[i])
        return List(temp_list)

    def __mul__(self, other):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise UnequalLengthException
            for i in range(len(self)):
                temp_list.append(self[i] * other[i])
            return List(temp_list)
        for i in range(len(self)):
            temp_list.append(self[i] * other)
        return List(temp_list)

    __imul__ = __mul__

    def __rmul__(self, other):
        temp_list = []
        for i in range(len(self)):
            temp_list.append(other * self[i])
        return List(temp_list)

    def __truediv__(self, other):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise UnequalLengthException
            for i in range(len(self)):
                temp_list.append(self[i] / other[i])
            return List(temp_list)
        for i in range(len(self)):
            temp_list.append(self[i] / other)
        return List(temp_list)

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        temp_list = []
        for i in range(len(self)):
            temp_list.append(other / self[i])
        return List(temp_list)

    def evalf(self, n=15, **options):
        temp_list = []
        for item in self:
            temp_list.append(sympy.N(item, n, **options))
        print(temp_list[0])
        return List(temp_list)

    def __pow__(self, other, modulo=None):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise UnequalLengthException
            if modulo:
                for i in range(len(self)):
                    temp_list.append(pow(self[i], other[i]) % modulo)
            else:
                for i in range(len(self)):
                    temp_list.append(pow(self[i], other[i]))
            return List(temp_list)
        for i in range(len(self)):
            temp_list.append(pow(self[i], other))
        return List(temp_list)

    __ipow__ = __pow__

    def __rpow__(self, other):
        temp_list = []
        for i in range(len(self)):
            temp_list.append(pow(other, self[i]))
        return List(temp_list)


class Rule(tuple):
    # TODO: Pretty and LaTeX printing
    def __init__(self, a, b):
        self.lhs = a
        self.rhs = b

    def __new__(cls, a, b):
        return super(Rule, cls).__new__(Rule, (a, b))

    def __repr__(self):
        return f'{self[0]} -> {self[1]}'

    def __str__(self):
        return self.__repr__()
