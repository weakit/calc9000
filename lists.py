import sympy as s
# TODO: Nothing
# TODO: check others


class RuleException(Exception):
    pass


class ListException(Exception):
    pass


class List(s.Basic):  # probably a bad idea
    # TODO: Pretty and LaTeX printing
    @staticmethod
    def create(*args):
        return List(args)

    def __init__(self, x=None):
        if x is None:
            self.value = []
        else:
            self.value = list(x)

    def __getitem__(self, x):
        return self.value.__getitem__(x)

    def __setitem__(self, key, value):
        self.value[key] = value

    def __iter__(self):
        return self.value.__iter__()

    def __len__(self):
        return self.value.__len__()

    def __add__(self, other):
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(self)):
                self[i] = self[i] + other[i]
            return self
        for i in range(len(self)):
            self[i] = self[i] + other
        return self

    def __radd__(self, other):
        for i in range(len(self)):
            self[i] = other + self[i]
        return self

    def __sub__(self, other):
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(self)):
                self[i] = self[i] - other[i]
            return self
        for i in range(len(self)):
            self[i] = self[i] - other
        return self

    def __rsub__(self, other):
        for i in range(len(self)):
            self[i] = other - self[i]
        return self

    def __mul__(self, other):
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(self)):
                self[i] = self[i] * other[i]
            return self
        for i in range(len(self)):
            self[i] = self[i] * other
        return self

    def __rmul__(self, other):
        for i in range(len(self)):
            self[i] = other * self[i]
        return self

    def __truediv__(self, other):
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(self)):
                self[i] = self[i] / other[i]
            return self
        for i in range(len(self)):
            self[i] = self[i] / other
        return self

    def __rtruediv__(self, other):
        for i in range(len(self)):
            self[i] = other / self[i]
        return self

    def evalf(self, n=15, **options):
        for i in range(len(self)):
            self[i] = s.N(self[i], n, **options)
        return self

    def __pow__(self, other, modulo=None):
        if isinstance(other, List):
            if len(other) != len(self):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(self)):
                self[i] = pow(self[i], other[i]) % modulo
            return self
        for i in range(len(self)):
            self[i] = pow(self[i], other) % modulo
        return self

    def __rpow__(self, other):
        for i in range(len(self)):
            self[i] = pow(other, self[i])
        return self

    def __hash__(self):
        return tuple(self.value).__hash__()

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
        return List(list(x) + list(y))

    def append(self, *x):
        self.value += list(x)


class Rule(s.AtomicExpr):
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
