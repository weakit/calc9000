from . import s, p, p_str, extra_precision
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


def test_head():
    assert p_str('Head[f[a, b]]') == 'f'
    assert p_str('Head[{a, b, c}]') == 'List'
    assert p_str('Head[45]') == 'Integer'
    assert p_str('Head[x]') == 'Symbol'
    # TODO: Compound Functions
    # assert p_str('Head[f[x][y][z]]') == 'f[x][y]'


def test_min():
    assert p('Min[9, 2]') == 2
    assert p('Min[{4, 1, 7, 2}]') == 1
    assert p_str('Min[5.56, -4.8, 7.3]').startswith('-4.8')
    assert p_str(f'N[Min[1/7, 4/5, 1], 50-{extra_precision}]') == '0.14285714285714285714285714285714285714285714285714'
    assert p('Min[{{-1, 0, 1, 2}, {0, 2, 4, 6}, {-3, -2, -1, 0}}]') == -3
    assert p('Min[Infinity, 5]') == 5
    assert p('Min[-Infinity, -5]') == -1 * s.oo
    assert p_str('Min[{E, Pi, 5}]') == 'E'


def test_max():
    assert p('Max[9, 2]') == 9
    assert p('Max[{4, 1, 7, 2}]') == 7
    assert p_str('Max[5.56, -4.8, 7.3]').startswith('7.3')
    assert p('N[Max[1/7, 4/5, 1], 50-4]') == 1
    assert p('Max[{{-1, 0, 1, 2}, {0, 2, 4, 6}, {-3, -2, -1, 0}}]') == 6
    assert p('Max[Infinity, 5]') == s.oo
    assert p('Max[-Infinity, -5]') == -5
    assert p('Max[{E, Pi, 5}]') == 5


def test_total():
    assert p('Total[{a, b, c, d}]') == p('a + b + c + d')
    assert p('Total[{Pi, 1, 4, E, 7, 8}]') == p('20 + E + Pi')
    assert p('Total[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]') == p('{12, 15, 18}')
    assert p('Total[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {2}]') == p('{6, 15, 24}')
    assert p('Total[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 2]') == 45
    assert p('Total[{x^2, 3 x^3, 1}]') == p('3x^3 + x^2 + 1')


def test_accumulate():
    assert p('Accumulate[{a, b, c, d}]') == p('{a, a+b, a+b+c, a+b+c+d}')
    assert p('Accumulate[{{a, b}, {c, d}, {e, f}}]') == p('{{a, b}, {a+c, b+d}, {a+c+e, b+d+f}}')
    assert p('Accumulate[f[a, b, c, d]]') == p('f[a, a+b, a+b+c, a+b+c+d]')
    assert p('Accumulate[Range[10]]') == p('{1, 3, 6, 10, 15, 21, 28, 36, 45, 55}')


def test_range():
    assert p('Range[4]') == p('{1, 2, 3, 4}')
    assert p('Range[x, x + 4]') == p('{x, x+1, x+2, x+3, x+4}')
    assert p('Range[1, 10, 2]') == p('{1, 3, 5, 7, 9}')
    assert p('Range[10, 1, -1]') == p('{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}')
    assert p('Range[0, 10, Pi]') == p('{0, Pi, 2 Pi, 3 Pi}')
    assert p('Range[a, b, (b - a)/4]') == p('{a, a+(-a+b)/4, a+(-a+b)/2, a+(3*(-a+b))/4, b} ')
    assert p('Range[{5, 2, 6, 3}]') == p('{{1, 2, 3, 4, 5}, {1, 2}, {1, 2, 3, 4, 5, 6}, {1, 2, 3}}')
    assert p('Range[-4, 9, 3]') == p('{-4,-1,2,5,8} ')
    assert p('Range[1, 10, 1/10]') == \
           p("{1, 11/10, 6/5, 13/10, 7/5, 3/2, 8/5, 17/10, 9/5, 19/10, 2, 21/10, 11/5, 23/10,"
             "12/5, 5/2, 13/5, 27/10, 14/5, 29/10, 3, 31/10, 16/5, 33/10, 17/5, 7/2, 18/5, 37/10,"
             "19/5, 39/10, 4, 41/10, 21/5, 43/10, 22/5, 9/2, 23/5, 47/10, 24/5, 49/10, 5, 51/10,"
             "26/5, 53/10, 27/5, 11/2, 28/5, 57/10, 29/5, 59/10, 6, 61/10, 31/5, 63/10, 32/5,"
             "13/2, 33/5, 67/10, 34/5, 69/10, 7, 71/10, 36/5, 73/10, 37/5, 15/2, 38/5, 77/10,"
             "39/5, 79/10, 8, 81/10, 41/5, 83/10, 42/5, 17/2, 43/5, 87/10, 44/5, 89/10, 9, 91/10,"
             "46/5, 93/10, 47/5, 19/2, 48/5, 97/10, 49/5, 99/10, 10} ")  # big test


# def test_permutations():
#     assert p('Permutations[{a, b, c}]') == p('{{a,b,c},{a,c,b},{b,a,c},{b,c,a},{c,a,b},{c,b,a}}')
#     assert p('Permutations[{a, b, c, d}, {3}]') == \
#            p("{{a,b,c},{a,b,d},{a,c,b},{a,c,d},{a,d,b},{a,d,c},{b,a,c},{b,a,d},{b,c,a},{b,c,d},"
#              "{b,d,a},{b,d,c},{c,a,b},{c,a,d},{c,b,a},{c,b,d},{c,d,a},{c,d,b},{d,a,b},{d,a,c},"
#              "{d,b,a},{d,b,c},{d,c,a},{d,c,b}} ")
#     assert p('Permutations[{a, a, b}]') == p('{{a,a,b},{a,b,a},{b,a,a}}')
#     assert p('Permutations[{x, x^2, x + 1}]') == \
#            p("{{x, x^2, 1 + x}, {x, 1 + x, x^2}, {x^2, x, 1 + x}, {x^2, 1 + x, x},"
#              "{1 + x, x, x^2}, {1 + x, x^2, x}}")
#     assert p('Permutations[Range[3], All]') == \
#            p("{{},{1},{2},{3},{1,2},{1,3},{2,1},{2,3},{3,1},{3,2},{1,2,3},{1,3,2},"
#              "{2,1,3},{2,3,1},{3,1,2},{3,2,1}} ")
#     assert p('Permutations[Range[4], {4, 0, -2}]') == \
#            p("{{1,2,3,4},{1,2,4,3},{1,3,2,4},{1,3,4,2},{1,4,2,3},{1,4,3,2},{2,1,3,4},"
#              "{2,1,4,3},{2,3,1,4},{2,3,4,1},{2,4,1,3},{2,4,3,1},{3,1,2,4},{3,1,4,2},"
#              "{3,2,1,4},{3,2,4,1},{3,4,1,2},{3,4,2,1},{4,1,2,3},{4,1,3,2},{4,2,1,3},"
#              "{4,2,3,1},{4,3,1,2},{4,3,2,1},{1,2},{1,3},{1,4},{2,1},{2,3},{2,4},{3,1},"
#              "{3,2},{3,4},{4,1},{4,2},{4,3},{}} ")
