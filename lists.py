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
        if isinstance(x, slice):
            return List(self.value.__getitem__(x))
        else:
            return self.value.__getitem__(x)

    def __setitem__(self, key, value):
        self.value[key] = value

    def __iter__(self):
        return self.value.__iter__()

    def __len__(self):
        return self.value.__len__()

    def __add__(self, other):
        new = List(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] + other[i]
            return new
        for i in range(len(new)):
            new[i] = new[i] + other
        return new

    def __radd__(self, other):
        new = List(self)
        for i in range(len(new)):
            new[i] = other + new[i]
        return new

    def __sub__(self, other):
        new = List(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] - other[i]
            return new
        for i in range(len(new)):
            new[i] = new[i] - other
        return new

    def __rsub__(self, other):
        new = List(self)
        for i in range(len(new)):
            new[i] = other - new[i]
        return new

    def __mul__(self, other):
        new = List(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] * other[i]
            return new
        for i in range(len(new)):
            new[i] = new[i] * other
        return new

    def __rmul__(self, other):
        new = List(self)
        for i in range(len(new)):
            new[i] = other * new[i]
        return new

    def __truediv__(self, other):
        new = List(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] / other[i]
            return new
        for i in range(len(new)):
            new[i] = new[i] / other
        return new

    def __rtruediv__(self, other):
        new = List(self)
        for i in range(len(new)):
            new[i] = other / new[i]
        return new

    def evalf(self, n=15, **options):
        new = List(self)
        for i in range(len(new)):
            new[i] = s.N(new[i], n, **options)
        return new

    def __pow__(self, other):
        new = List(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = pow(new[i], other[i])
            return new
        for i in range(len(new)):
            new[i] = pow(new[i], other)
        return new

    def __rpow__(self, other):
        new = List(self)
        for i in range(len(new)):
            new[i] = pow(other, new[i])
        return new

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
