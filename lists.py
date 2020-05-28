import sympy
# TODO: Nothing
# TODO: check others


class RuleException(Exception):
    pass


class ListException(Exception):
    pass


class List(sympy.Basic):  # probably a bad idea
    # TODO: Pretty and LaTeX printing
    # is_number = True

    def __init__(self, x=()):
        self.value = tuple(x)

    def __getitem__(self, x):
        return self.value.__getitem__(x)

    def __iter__(self):
        return self.value.__iter__()

    def __len__(self):
        return self.value.__len__()

    def __add__(self, other):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
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
                raise ListException("Lists are of Unequal Length")
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
                raise ListException("Lists are of Unequal Length")
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
                raise ListException("Lists are of Unequal Length")
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
        return List(temp_list)

    def __pow__(self, other, modulo=None):
        temp_list = []
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
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

    def __hash__(self):
        return self.value.__hash__()

    def __repr__(self):
        if not self.value:
            return '{}'
        string = '{' + repr(self.value[0])
        for value in self.value[1:]:
            string += ', ' + repr(value)
        return string + '}'

    def __str__(self):
        if not self.value:
            return '{}'
        string = '{' + str(self.value[0])
        for value in self.value[1:]:
            string += ', ' + str(value)
        return string + '}'

    @classmethod
    def concat(cls, x, y):
        if isinstance(x, List):
            x = x.value
        if isinstance(y, List):
            y = y.value
        return List(tuple(x) + tuple(y))

    def append(self, x):
        self.value += (x,)


class Rule(sympy.Expr):
    # TODO: Pretty and LaTeX printing
    def __init__(self, a, b):
        self.lhs = a
        self.rhs = b

    def __getitem__(self, item):
        if item == 0:
            return self.lhs
        if item == 1:
            return self.rhs
        raise RuleException

    def __iter__(self):
        return (self.lhs, self.rhs).__iter__()

    def __repr__(self):
        return f'{self[0]} -> {self[1]}'

    def __str__(self):
        return self.__repr__()
