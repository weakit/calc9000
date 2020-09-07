import sympy as s
# TODO: check others
# TODO: LaTeX printing
# TODO: Rule Delayed (@property)


class RuleException(Exception):
    pass


class ListException(Exception):
    pass


class List(s.Basic):
    @staticmethod
    def create(x):
        return List(*x)

    def __init__(self, *args):
        if args:
            self.value = list(filter(s.Symbol('Nothing').__ne__, args))
        else:
            self.value = []

    def __getitem__(self, x):
        if isinstance(x, slice):
            return List(*self.value.__getitem__(x))
        return self.value.__getitem__(x)

    def __setitem__(self, key, value):
        self.value[key] = value
        self.value = list(filter(s.Symbol('Nothing').__ne__, self.value))

    def __iter__(self):
        return self.value.__iter__()

    def __len__(self):
        return self.value.__len__()

    def __add__(self, other):
        if isinstance(other, List):
            if len(other) != len(self.value):
                raise ListException("Lists are of Unequal Length")
            return List(*(self.value[i] + other[i] for i in range(len(self.value))))
        return List(*(x + other for x in self.value))

    def __radd__(self, other):
        return List(*(other + x for x in self.value))

    def __sub__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            return List(*(self.value[i] - other[i] for i in range(len(self.value))))
        return List(*(x - other for x in self.value))

    def __rsub__(self, other):
        return List(*(other - x for x in self.value))

    def __mul__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            return List(*(self.value[i] * other[i] for i in range(len(self.value))))
        return List(*(x * other for x in self.value))

    def __rmul__(self, other):
        return List(*(other * x for x in self.value))

    def __truediv__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            return List(*(self.value[i] / other[i] for i in range(len(self.value))))
        return List(*(x / other for x in self.value))

    def __rtruediv__(self, other):
        return List(*(other / x for x in self.value))

    def evalf(self, n=15, **options):
        new = list(self)
        for i in range(len(new)):
            new[i] = s.N(new[i], n, **options)
        return List(*new)

    def __pow__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException("Lists are of Unequal Length")
            return List(*(pow(self.value[i], other[i]) for i in range(len(self.value))))
        return List(*(pow(x, other) for x in self.value))

    def __rpow__(self, other):
        return List(*(pow(other, x) for x in self.value))

    def __hash__(self):
        return (List, *self.value).__hash__()

    def __eq__(self, other):
        if isinstance(other, List):
            return self.value == other.value
        return False

    def __repr__(self):
        return self.__str__()

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
        self.value += list(filter(s.Symbol('Nothing').__ne__, x))


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
        return self.__str__()

    def __str__(self):
        return f'{self[0]} -> {self[1]}'

    @classmethod
    def from_dict(cls, d, head=lambda *x: tuple(x)):
        return head(*(Rule(x, d[x]) for x in d))


class Tag(s.Symbol):

    @property
    def symbol(self):
        return self.name.split('::')[0]

    @property
    def tag(self):
        return self.name.split('::')[-1]


class Span(s.AtomicExpr):
    # accept only 3 arguments to keep it simple
    def __init__(self, a=None, b=None, c=None):
        if a is None:
            a = 1
        if b is None:
            b = s.Symbol('All')

        self.a, self.b, self.c = a, b, c

    def __str__(self):
        if self.c:
            return f'{self.a};;{self.b};;{self.c}'
        return f'{self.a};;{self.b}'

    def __repr__(self):
        return self.__str__()

    def take_spec(self):
        if self.b == s.Symbol('All'):
            b = -1
        else:
            b = self.b
        return List(self.a, b, self.c or 1)


class String(s.AtomicExpr):
    def __init__(self, x: str):
        self.value = str(x)

    def __hash__(self):
        self.value.__hash__()

    def __str__(self):
        return f'"{self.value}"'

    def __repr__(self):
        return self.__str__()
