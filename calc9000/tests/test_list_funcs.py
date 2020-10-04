from . import s, p
from calc9000.custom import List


def test_part():
    a, b, c, d, e, f, g, h, i = s.var('a b c d e f g h i')
    assert p('Part[{a, b, c, d, e}, 3]') == c
    assert p('{a, b, c, d, e, f}[[3]]') == c
    assert p('{{a, b, c}, {d, e, f}}[[1]][[2]]') == b
    assert p('{{a, b, c}, {d, e, f}}[[1, 2]]') == b
    assert p('{{a, b, c}, {d, e, f}, {g, h, i}}[[{1, 3}]]') == List(List(a, b, c), List(g, h, i))
    assert p('{{a, b, c}, {d, e, f}, {g, h, i}}[[{1, 3}, {2, 3}]]') == List(List(b, c), List(h, i))
    assert p('{{a, b, c}, {d, e, f}, {g, h, i}}[[2, 3]]') == f
    assert p('{{a, b, c}, {d, e, f}, {g, h, i}}[[2]]') == List(d, e, f)
    assert p('{{a, b, c}, {d, e, f}, {g, h, i}}[[All, 2]]') == List(b, e, h)
    assert p('{a, b, c, d, e, f}[[-2]]') == e
    assert p('{a, b, c, d, e, f}[[{1, 3, 1, 2, -1, -1}]]') == List(a, c, a, b, f, f)
    assert p('{a, b, c, d, e, f}[[2 ;; 4]]') == List(b, c, d)
    assert p('{a, b, c, d, e, f}[[1 ;; -3]]') == p('{a, b, c, d, e, f}[[;; -3]]') == List(a, b, c, d)
    assert p('{a, b, c, d, e, f, g, h, i, j}[[3 ;; -3 ;; 2]]') == List(c, e, g)
    assert p('{a, b, c, d, e, f, g, h, i, j}[[;; ;; 2]]') == List(a, c, e, g, i)

    assert p('f[g[a, b], g[c, d]][[2, 1]]') == c
    assert p('(1 + 2 a^2 + b^2)[[2]]') == b ** 2
    assert p('{a -> c, b -> d}[[1, 2]]') == c
    assert p('(a/b)[[2]]') == 1/b
    assert p('{a, b, c}[[0]]') == p('List')
    assert p('f[a, b, c][[{2, 3}]] == f[b, c]')
    assert p('f[g[a, b], h[c, d]][[{1, 2}, {2}]] == f[g[b], h[d]]')


def test_take():
    a, b, c, d, e, f, t, u = s.var('a b c d e f t u')
    assert p('Take[{a, b, c, d, e, f}, 4]') == List(a, b, c, d)
    assert p('Take[{a, b, c, d, e, f}, -3]') == List(d, e, f)
    assert p('Take[{a, b, c, d, e, f}, {2, -2}]') == List(b, c, d, e)
    assert p('Take[{a, b, c, d, e, f}, {1, -1, 2}]') == List(a, c, e)
    assert p('Take[{a, b, c, d, e}, -1]') == List(e)
    assert p('Take[{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}}, 2]') == \
           List(List(11, 12, 13), List(21, 22, 23))
    assert p('Take[{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}}, All, 2]') == \
           List(List(11, 12), List(21, 22), List(31, 32))
    assert p('Take[{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}}, 2, -1]') == \
           List(List(13), List(23))
    m = "{{11,12,13,14,15},{21,22,23,24,25},{31,32,33,34,35},{41,42,43,44,45},{51,52,53,54,55}}"
    assert p('Take[' + m + ', {2, 4}, {3, 5}]') == \
           List(List(23, 24, 25), List(33, 34, 35), List(43, 44, 45))
    assert p('Take[' + m + ', {1, -1, 2}, {1, -1, 2}]') == p('{{11,13,15},{31,33,35},{51,53,55}}')
    assert p('Take[a + b + c + d + e + f, 3]') == a + b + c
    assert p('Take[{a + b + c, t + u + v, x + y + z}, 2, 2]') == List(a + b, t + u)


