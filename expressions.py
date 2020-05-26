import sympy as s
import references as r
from functools import reduce
import operator
from functions import Functions


def numeric(n):
    return s.Number(n)


def symbol(n):
    if n in r.refs.Constants.__dict__:
        return r.refs.Constants.__dict__[n]
    if n not in r.refs.Symbols.__dict__:
        r.refs.Symbols.__setattr__(n, s.Symbol(n))
    ret = r.refs.Symbols.__dict__[n]
    if ret != s.symbols(n) and isinstance(ret, s.Expr):
        ret = ret.subs(vars(r.refs.Symbols))
    return ret


def function(n):
    return Functions.call(str(n[0]), *n[1:])


def plus(n):
    return reduce(operator.add, n)


def subtract(n):
    return reduce(operator.sub, n)


def times(n):
    return reduce(operator.mul, n)


def dot(n):
    return Functions.call("Dot", *n)


def divide(n):
    return n[0] / times(n[1:])


def factorial(n):
    return Functions.call("Factorial", n[0])


def power(n):
    return reduce(operator.pow, n)


def positive(n):
    return 1 * n[0]


def negative(n):
    return -1 * n[0]


def out(n):
    try:
        return Functions.call("Out", int(n[-1]))
    except ValueError:
        return Functions.call("Out", r.refs.Line - len(n))


def relations(n):
    relation = True
    for x in range(1, len(n), 2):
        relation = relation & s.Rel(n[x - 1], n[x + 1], n[x])
    return relation


def assign(n):
    for x in n[1:-1]:
        Functions.call("Set", x, n[-1])
    return Functions.call("Set", n[0], n[-1])


def unset(n):
    return Functions.call("Unset", n)


def And(n):
    return Functions.call("And", *n)
