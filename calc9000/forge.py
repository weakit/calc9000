import sympy as s
from functools import reduce
import operator as op
from calc9000 import functions
from calc9000.datatypes import List, Rule, Tag, String
from lark import Tree


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
    'and_',
    'or_',
    'not_'
)

# TODO: remove unused functions


def numeric(n):
    return s.Number(n)


def float_(n: str):
    return s.Float(n, max(20, len(n.lstrip('0').replace('.', '')) + 4))


def symbol(n):
    return functions.get_symbol_value(n)


def tag(a, b):
    t = functions.get_tag_value(a, b)
    if isinstance(t, str):
        return String(t)
    return t


def lazy_tag(a, b):
    return Tag(f'{a}::{b}')


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
        return Functions.call('Out', -len(n))


def relations(n):
    relation = True
    for x in range(1, len(n), 2):
        relation = relation & s.Rel(n[x - 1], n[x + 1], n[x])
    return relation


def assign(n):
    # n = [pilot(x) for x in n]
    for x in n[1:-1]:
        Functions.call('Set', x, n[-1])
    return Functions.call('Set', n[0], n[-1])


def unset(n):
    return Functions.call('Unset', lazy(n))


def and_(n):
    return Functions.call('And', *n)


def or_(n):
    return Functions.call('Or', *n)


def not_(n):
    return Functions.call('Not', *n)


def part(n):
    return Functions.call('Part', *n)


def replace(n):
    return Functions.call('Subs', *n)


def delayed(n, f):
    return Functions.call('DelayedSet', f, *n)


# TODO: Part and Logical Operators

def lazy(tree: Tree):
    if not isinstance(tree, Tree):
        return tree
    if tree.data == 'symbol':
        return s.Symbol(tree.children[0])
    if tree.data in basic_ops:
        # TODO: Convert to hold
        return globals()[tree.data]([lazy(x) for x in tree.children])
    if tree.data == 'function':
        return Functions.lazy_call(str(tree.children[0].children[0]), *(lazy(x) for x in tree.children[1:]))
    if tree.data == 'list':
        return List(*(lazy(x) for x in tree.children))
    if tree.data == 'rule':
        return Rule(*(lazy(x) for x in tree.children))
    if tree.data == 'part':
        return Functions.lazy_call('Part', *(lazy(x) for x in tree.children))
    if tree.data == 'out':
        return out(tree.children)
    if tree.data == 'set':
        return Functions.lazy_call('Set', lazy(tree.children[0]), lazy(tree.children[1]))
    if tree.data == 'set_delayed':
        return Functions.lazy_call('SetDelayed', lazy(tree.children[0]), lazy(tree.children[1]))
    if tree.data == 'semicolon_statement':
        return Functions.lazy_call('SemicolonStatement', *(lazy(x) for x in tree.children))
    if tree.data == 'compound_statement':
        return Functions.lazy_call('CompoundExpression', *(lazy(x) for x in tree.children))
    if tree.data == 'RELATIONAL':
        return str(tree.children[0])
    if tree.data == 'relation':
        return relations([lazy(x) for x in tree.children])
    if tree.data == 'tag':
        return lazy_tag(*(lazy(x) for x in tree.children))


def operate(tree: Tree):
    if not isinstance(tree, Tree):
        return tree
    if tree.data == 'symbol':
        return symbol(tree.children[0])
    if tree.data in basic_ops:
        return globals()[tree.data]([operate(x) for x in tree.children])
    if tree.data == 'function':
        name = str(tree.children[0].children[0])
        if Functions.is_explicit(name):
            return Functions.call(name, *(lazy(x) for x in tree.children[1:]))
        return Functions.call(name, *(operate(x) for x in tree.children[1:]))
    if tree.data == 'list':
        return List(*(operate(x) for x in tree.children))
    if tree.data == 'rule':
        return Rule(*(operate(x) for x in tree.children))
    if tree.data == 'part':
        return Functions.call('Part', *(operate(x) for x in tree.children))
    if tree.data == 'out':
        return out(tree.children)
    if tree.data == 'set':
        return Functions.call('Set', lazy(tree.children[0]), operate(tree.children[1]))
    if tree.data == 'set_delayed':
        return Functions.call('SetDelayed', lazy(tree.children[0]), lazy(tree.children[1]))
    if tree.data == 'replace':  # see ReplaceAll
        return Functions.call('Subs', *(operate(x) for x in tree.children))
    if tree.data == 'unset':
        return Functions.call('Unset', lazy(tree.children[0]))
    if tree.data == 'semicolon_statement':
        return Functions.call('SemicolonStatement', *(lazy(x) for x in tree.children))
    if tree.data == 'compound_statement':
        return Functions.call('CompoundExpression', *(lazy(x) for x in tree.children))
    if tree.data == 'RELATIONAL':
        return str(tree.children[0])
    if tree.data == 'relation':
        return relations([operate(x) for x in tree.children])
    if tree.data == 'tag':
        return tag(*(operate(x) for x in tree.children))
    return tree
