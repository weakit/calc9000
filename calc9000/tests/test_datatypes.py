from . import p, f
from calc9000.datatypes import List, Rule


def test_list():
    assert p('{1, 2, 3}') == List(1, 2, 3)
    d = p('{1, 2, 3, 4, 5}')
    assert d[0] == 1
    d[0] = 2
    assert d[0] == 2
    assert p('{1, 2} + {3, 4}') == List(4, 6)
    assert p('{1, 2} / {4, 3}') == List(f(1, 4), f(2, 3))
    assert p('{1, 2, 3, 4} + 1') == List(2, 3, 4, 5)
    assert p('N[{1, 2, 3}]') == List(1, 2, 3)
    assert p('{1, 2, 3} ^ 2') == List(1, 4, 9)
    assert p('{1, 2, 3} ^ {3, 2, 1}') == List(1, 4, 3)
    assert p('{1, 2, 3, x}') == p('{2 - 1, 2, 2 + 1, x}')
    assert list(p('{1, 2, 3}')) == [1, 2, 3]


def test_rule():
    assert p('1 -> 2') == Rule(1, 2)
    assert p('2 -> 3').lhs == 2
    r = p('1 -> 2')
    r.lhs += 1
    assert p('2 -> 2') == r
    assert (r.lhs, r.rhs) == (r[0], r[1])
    assert list(r) == [2, 2]
