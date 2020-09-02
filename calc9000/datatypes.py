import sympy as s
# TODO: check others
# TODO: LaTeX printing


class RuleException(Exception):
    pass


class ListException(Exception):
    pass


class List(s.Basic):
    @staticmethod
    def create(x):
        return List(*x)

    def __init__(self, *args):
        if args is None:
            self.value = []
        else:
            self.value = list(filter(s.Symbol('Nothing').__ne__, args))

    def __getitem__(self, x):
        if isinstance(x, slice):
            return List(*self.value.__getitem__(x))
        else:
            return self.value.__getitem__(x)

    def __setitem__(self, key, value):
        self.value[key] = value
        self.value = list(filter(s.Symbol('Nothing').__ne__, self.value))

    def __iter__(self):
        return self.value.__iter__()

    def __len__(self):
        return self.value.__len__()

    def __add__(self, other):
        new = list(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] + other[i]
            return List(*new)
        for i in range(len(new)):
            new[i] = new[i] + other
        return List(*new)

    def __radd__(self, other):
        new = list(self)
        for i in range(len(new)):
            new[i] = other + new[i]
        return List(*new)

    def __sub__(self, other):
        new = list(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] - other[i]
            return List(*new)
        for i in range(len(new)):
            new[i] = new[i] - other
        return List(*new)

    def __rsub__(self, other):
        new = list(self)
        for i in range(len(new)):
            new[i] = other - new[i]
        return List(*new)

    def __mul__(self, other):
        new = list(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] * other[i]
            return List(*new)
        for i in range(len(new)):
            new[i] = new[i] * other
        return List(*new)

    def __rmul__(self, other):
        new = list(self)
        for i in range(len(new)):
            new[i] = other * new[i]
        return List(*new)

    def __truediv__(self, other):
        new = list(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = new[i] / other[i]
            return List(*new)
        for i in range(len(new)):
            new[i] = new[i] / other
        return List(*new)

    def __rtruediv__(self, other):
        new = list(self)
        for i in range(len(new)):
            new[i] = other / new[i]
        return List(*new)

    def evalf(self, n=15, **options):
        new = list(self)
        for i in range(len(new)):
            new[i] = s.N(new[i], n, **options)
        return List(*new)

    def __pow__(self, other):
        new = list(self)
        if isinstance(other, List):
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            for i in range(len(new)):
                new[i] = pow(new[i], other[i])
            return List(*new)
        for i in range(len(new)):
            new[i] = pow(new[i], other)
        return List(*new)

    def __rpow__(self, other):
        new = list(self)
        for i in range(len(new)):
            new[i] = pow(other, new[i])
        return List(*new)

    def __hash__(self):
        return (List, *self.value).__hash__()

    def __eq__(self, other):
        if isinstance(other, List):
            return self.value == other.value
        return False

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

    def concat(self, y):
        x = self
        if isinstance(self, List):
            x = x.value
        if isinstance(y, List):
            y = y.value
        return List.create(list(x) + list(y))

    def append(self, *x):
        self.value += list(x)


class Rule(s.AtomicExpr):
    def __init__(self, a, b):
        self.lhs = a
        self.rhs = b
        self._args = (a, b)

    def __getitem__(self, item):
        if item == 0:
            return self.lhs
        if item == 1:
            return self.rhs
        raise RuleException

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.lhs == other.lhs and self.rhs == other.rhs
        return False

    def __hash__(self):
        return (Rule, self.lhs, self.rhs).__hash__()

    def __iter__(self):
        return (self.lhs, self.rhs).__iter__()

    def __repr__(self):
        return f'{self[0]} -> {self[1]}'

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dict(cls, d, head=lambda *x: tuple(x)):
        return head(*(Rule(x, d[x]) for x in d))
