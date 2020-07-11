import sympy as s
import references as r
from functools import reduce
import operator as op
import functions
from lark import Tree
from datatypes import List, Rule


Functions = functions.Functions

basic_ops = (
    'plus',
    'subtract',
    'times',
    'divide',
    'power',
    'negative',
    'positive',
    'dot',
    'factorial',
)


def numeric(n):
    return s.Number(n)


def symbol(n):
    if n in r.refs.Constants.__dict__:
        return r.refs.Constants.__dict__[n]
    if n not in r.refs.Symbols:
        r.refs.Symbols[n] = s.Symbol(n)
    ret = r.refs.Symbols[n]
    if ret != s.symbols(n) and isinstance(ret, s.Expr):
        ret = ret.subs(vars(r.refs.Symbols))
    return ret


def function(n):
    return Functions.call(str(n[0]), *n[1:])


def unset_function(n):
    return s.Function(str(n[0]))(*n[1:])


def plus(n):
    return reduce(op.add, n)


def subtract(n):
    return reduce(op.sub, n)


def times(n):
    return reduce(op.mul, n)


def dot(n):
    return Functions.call('Dot', *n)


def divide(n):
    return n[0] / times(n[1:])


def factorial(n):
    return Functions.call('Factorial', n[0])


def power(n):
    return reduce(op.pow, n)


def positive(n):
    return 1 * n[0]


def negative(n):
    return -1 * n[0]


def out(n):
    try:
        return Functions.call('Out', int(n[-1]))
    except ValueError:
        return Functions.call('Out', r.refs.Line - len(n))


def relations(n):
    relation = True
    for x in range(1, len(n), 2):
        relation = relation & s.Rel(n[x - 1], n[x + 1], n[x])
    return relation


def assign(n):
    n = [pilot(x) for x in n]
    for x in n[1:-1]:
        Functions.call('Set', x, n[-1])
    return Functions.call('Set', n[0], n[-1])


def unset(n):
    return Functions.call('Unset', pilot(n))


def _and(n):
    return Functions.call('And', *n)


def part(n):
    return Functions.call('Part', *n)


def replace(n):
    return Functions.call('Subs', *n)


def delayed(n, f):
    return functions.DelayedSet(f, *n)


# TODO: Part, Replace and Logical Operators

def pilot(tree: Tree):
    if not isinstance(tree, Tree):
        return tree
    if tree.data == 'symbol':
        return s.Symbol(tree.children[0])
    if tree.data in basic_ops:
        return globals()[tree.data]([pilot(x) for x in tree.children])
    if tree.data == 'list':
        return List(*(pilot(x) for x in tree.children))
    if tree.data == 'rule':
        return Rule(*(pilot(x) for x in tree.children))
    if tree.data == 'part':
        return Functions.pilot_call('Part', *(pilot(x) for x in tree.children))
    if tree.data == 'RELATIONAL':
        return str(tree.children[0])
    if tree.data == 'relation':
        return relations([pilot(x) for x in tree.children])
    if tree.data == 'function':
        return Functions.pilot_call(str(tree.children[0].children[0]), *(pilot(x) for x in tree.children[1:]))


def operate(tree: Tree):
    if not isinstance(tree, Tree):
        return tree
    if tree.data == 'symbol':
        return symbol(tree.children[0])
    if tree.data in basic_ops:
        return globals()[tree.data]([operate(x) for x in tree.children])
    if tree.data == 'list':
        return List(*(operate(x) for x in tree.children))
    if tree.data == 'rule':
        return Rule(*(operate(x) for x in tree.children))
    if tree.data == 'part':
        return Functions.call('Part', *(operate(x) for x in tree.children))
    if tree.data == 'out':
        return out(tree.children)
    if tree.data == 'RELATIONAL':
        return str(tree.children[0])
    if tree.data == 'relation':
        return relations([operate(x) for x in tree.children])
    if tree.data == 'function':
        name = str(tree.children[0].children[0])
        if Functions.not_normal(name):
            return Functions.call(name, *(pilot(x) for x in tree.children[1:]))
        return Functions.call(name, *(operate(x) for x in tree.children[1:]))
    return tree
