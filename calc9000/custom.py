import sympy as s
# TODO: check others
# TODO: LaTeX printing
# TODO: Rule Delayed (@property)


class CustomException(Exception):
    message_symbol = None
    message_tag = None


class RuleException(CustomException):
    message_symbol = 'Rule'


class ListException(CustomException):
    message_symbol = 'List'


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

    def __contains__(self, item):
        return self.value.__contains__(item)

    # Lists have operations for internal functionality.

    def __add__(self, other):
        if isinstance(other, List):
            if len(other) != len(self.value):
                raise ListException(f'{self} and {other} have incompatible shapes.')
            return List(*(self.value[i] + other[i] for i in range(len(self.value))))
        return List(*(x + other for x in self.value))

    def __radd__(self, other):
        return List(*(other + x for x in self.value))

    def __sub__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException(f'{self} and {other} have incompatible shapes.')
            return List(*(self.value[i] - other[i] for i in range(len(self.value))))
        return List(*(x - other for x in self.value))

    def __rsub__(self, other):
        return List(*(other - x for x in self.value))

    def __mul__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException(f'{self} and {other} have incompatible shapes.')
            return List(*(self.value[i] * other[i] for i in range(len(self.value))))
        return List(*(x * other for x in self.value))

    def __rmul__(self, other):
        return List(*(other * x for x in self.value))

    def __truediv__(self, other):
        if isinstance(other, List):
            new = list(self)
            if len(other) != len(new):
                raise ListException(f'{self} and {other} have incompatible shapes.')
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
                raise ListException(f'{self} and {other} have incompatible shapes.')
            return List(*(pow(self.value[i], other[i]) for i in range(len(self.value))))
        return List(*(pow(x, other) for x in self.value))

    def __rpow__(self, other):
        return List(*(pow(other, x) for x in self.value))

    def __hash__(self):
        return (List, *self.value).__hash__()

    def _eval_Eq(self, other):
        if isinstance(other, List):
            if len(self.value) != len(other.value):
                return False
        return None

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
        return List.create(self.value + list(y))

    # TODO: remove append from code
    def append(self, *x):
        """
        DO NOT USE.
        """
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
        raise RuleException(f'Rule does not have part {item}')

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

    def atoms(self):
        a = (hasattr(self.a, 'atoms') and self.a.atoms()) or set()
        b = (hasattr(self.b, 'atoms') and self.b.atoms()) or set()
        c = (hasattr(self.c, 'atoms') and self.c.atoms()) or set()
        return a.union(b.union(c))


class String(s.AtomicExpr):
    def __init__(self, x: str):
        self.value = str(x)

    def __hash__(self):
        self.value.__hash__()

    def __str__(self):
        return f'"{self.value}"'

    def __repr__(self):
        return self.__str__()


class Primes(s.Set):
    # This won't work
    is_iterable = True
    is_empty = False
    _inf = s.Integer(2)
    _sup = s.S.Infinity
    is_finite_set = False

    def _contains(self, other):
        if not isinstance(other, s.Expr):
            return s.S.false
        return other.is_prime

    def __iter__(self):
        i = 1
        while True:
            yield s.prime(i)
            i += 1

    @property
    def _boundary(self):
        return s.Interval(2, s.S.Infinity)

    def _eval_is_subset(self, other):
        return s.Range(s.oo).is_subset(other)

    def _eval_is_superset(self, other):
        return s.Range(s.oo).is_superset(other)
