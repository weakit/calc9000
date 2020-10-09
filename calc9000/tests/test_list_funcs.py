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
    assert p('Range[1]') == p('{1}')
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


def test_permutations():
    assert p('Permutations[{a, b, c}]') == p('{{a,b,c},{a,c,b},{b,a,c},{b,c,a},{c,a,b},{c,b,a}}')
    assert p('Permutations[{a, b, c, d}, {3}]') == \
           p("{{a,b,c},{a,b,d},{a,c,b},{a,c,d},{a,d,b},{a,d,c},{b,a,c},{b,a,d},{b,c,a},{b,c,d},"
             "{b,d,a},{b,d,c},{c,a,b},{c,a,d},{c,b,a},{c,b,d},{c,d,a},{c,d,b},{d,a,b},{d,a,c},"
             "{d,b,a},{d,b,c},{d,c,a},{d,c,b}} ")
    assert p('Permutations[{a, a, b}]') == p('{{a,a,b},{a,b,a},{b,a,a}}')
    assert p('Permutations[{x, x^2, x + 1}]') == \
           p("{{x, x^2, 1 + x}, {x, 1 + x, x^2}, {x^2, x, 1 + x}, {x^2, 1 + x, x},"
             "{1 + x, x, x^2}, {1 + x, x^2, x}}")
    assert p('Permutations[Range[3], All]') == \
           p("{{},{1},{2},{3},{1,2},{1,3},{2,1},{2,3},{3,1},{3,2},{1,2,3},{1,3,2},"
             "{2,1,3},{2,3,1},{3,1,2},{3,2,1}} ")
    assert p('Permutations[Range[4], {4, 0, -2}]') == \
           p("{{1,2,3,4},{1,2,4,3},{1,3,2,4},{1,3,4,2},{1,4,2,3},{1,4,3,2},{2,1,3,4},"
             "{2,1,4,3},{2,3,1,4},{2,3,4,1},{2,4,1,3},{2,4,3,1},{3,1,2,4},{3,1,4,2},"
             "{3,2,1,4},{3,2,4,1},{3,4,1,2},{3,4,2,1},{4,1,2,3},{4,1,3,2},{4,2,1,3},"
             "{4,2,3,1},{4,3,1,2},{4,3,2,1},{1,2},{1,3},{1,4},{2,1},{2,3},{2,4},{3,1},"
             "{3,2},{3,4},{4,1},{4,2},{4,3},{}} ")
    assert p('Permutations[f[a, b, c]]') == \
           p('{f[a, b, c], f[a, c, b], f[b, a, c], f[b, c, a], f[c, a, b], f[c, b, a]} ')
    assert p('Permutations[Range[5], {1, 5, 3}]') == \
           p("{{1},{2},{3},{4},{5},{1,2,3,4},{1,2,3,5},{1,2,4,3},{1,2,4,5},{1,2,5,3},"
             "{1,2,5,4},{1,3,2,4},{1,3,2,5},{1,3,4,2},{1,3,4,5},{1,3,5,2},{1,3,5,4},"
             "{1,4,2,3},{1,4,2,5},{1,4,3,2},{1,4,3,5},{1,4,5,2},{1,4,5,3},{1,5,2,3},"
             "{1,5,2,4},{1,5,3,2},{1,5,3,4},{1,5,4,2},{1,5,4,3},{2,1,3,4},{2,1,3,5},"
             "{2,1,4,3},{2,1,4,5},{2,1,5,3},{2,1,5,4},{2,3,1,4},{2,3,1,5},{2,3,4,1},"
             "{2,3,4,5},{2,3,5,1},{2,3,5,4},{2,4,1,3},{2,4,1,5},{2,4,3,1},{2,4,3,5},"
             "{2,4,5,1},{2,4,5,3},{2,5,1,3},{2,5,1,4},{2,5,3,1},{2,5,3,4},{2,5,4,1},"
             "{2,5,4,3},{3,1,2,4},{3,1,2,5},{3,1,4,2},{3,1,4,5},{3,1,5,2},{3,1,5,4},"
             "{3,2,1,4},{3,2,1,5},{3,2,4,1},{3,2,4,5},{3,2,5,1},{3,2,5,4},{3,4,1,2},"
             "{3,4,1,5},{3,4,2,1},{3,4,2,5},{3,4,5,1},{3,4,5,2},{3,5,1,2},{3,5,1,4},"
             "{3,5,2,1},{3,5,2,4},{3,5,4,1},{3,5,4,2},{4,1,2,3},{4,1,2,5},{4,1,3,2},"
             "{4,1,3,5},{4,1,5,2},{4,1,5,3},{4,2,1,3},{4,2,1,5},{4,2,3,1},{4,2,3,5},"
             "{4,2,5,1},{4,2,5,3},{4,3,1,2},{4,3,1,5},{4,3,2,1},{4,3,2,5},{4,3,5,1},"
             "{4,3,5,2},{4,5,1,2},{4,5,1,3},{4,5,2,1},{4,5,2,3},{4,5,3,1},{4,5,3,2},"
             "{5,1,2,3},{5,1,2,4},{5,1,3,2},{5,1,3,4},{5,1,4,2},{5,1,4,3},{5,2,1,3},"
             "{5,2,1,4},{5,2,3,1},{5,2,3,4},{5,2,4,1},{5,2,4,3},{5,3,1,2},{5,3,1,4},"
             "{5,3,2,1},{5,3,2,4},{5,3,4,1},{5,3,4,2},{5,4,1,2},{5,4,1,3},{5,4,2,1},"
             "{5,4,2,3},{5,4,3,1},{5,4,3,2}}")


def test_table():
    assert p('Table[i, {i, 1, 10, 2}]') == p('{1,3,5,7,9}')
    assert p('Table[i^2, {i, 10}]') == p('{1,4,9,16,25,36,49,64,81,100}')
    assert p('Table[x^y, {x, 3}, {y, 4}]') == p('{{1,1,1,1},{2,4,8,16},{3,9,27,81}}')
    assert p('Table[f[i], {i, 0, 20, 2}]') == p('{f[0],f[2],f[4],f[6],f[8],f[10],f[12],f[14],f[16],f[18],f[20]}')
    assert p('Table[f[i], {i, 10, -5, -2}]') == p('{f[10],f[8],f[6],f[4],f[2],f[0],f[-2],f[-4]} ')
    assert p('Table[x, 10]') == p('{x,x,x,x,x,x,x,x,x,x} ')
    assert p('Table[10 i + j, {i, 4}, {j, 3}]') == p('{{11,12,13},{21,22,23},{31,32,33},{41,42,43}}')
    assert p('Table[i + j, {i, 3}, {j, i}]') == p('{{2},{3,4},{4,5,6}}')
    assert p('Table[Table[i + j, {j, i}], {i, 3}]') == p('{{2},{3,4},{4,5,6}} ')
    assert p('Table[10 i + j, {i, 5}, {j, i}]') == p('{{11},{21,22},{31,32,33},{41,42,43,44},{51,52,53,54,55}}')
    assert p('Table[100 i + 10 j + k, {i, 3}, {j, 2}, {k, 4}]') == \
           p("{{{111,112,113,114},{121,122,123,124}},{{211,212,213,214},"
             "{221,222,223,224}},{{311,312,313,314},{321,322,323,324}}} ")
    assert p('Table[Sqrt[x], {x, {1, 4, 9, 16}}]') == p('{1,2,3,4}')
    assert p('Table[2^x + x, {x, a, a + 5 n, n}]') == \
           p("{2^a + a, 2^(a + n) + a + n, 2^(a + 2*n) + a + 2*n, 2^(a + 3*n) + a + 3*n,"
             "2^(a + 4*n) + a + 4*n, 2^(a + 5*n) + a + 5*n}")
    assert p('Table[a[x]!, {a[x], 6}]') == p('{1,2,6,24,120,720}')
    assert p('Table[x[1]^2 + x[2]^2, {x[1], 3}, {x[2], 3}]') == p('{{2,5,10},{5,8,13},{10,13,18}}')
    assert p('Table[x, {x, 0, 10, 3}]') == p('{0,3,6,9}')
    from sympy.logic.boolalg import BooleanTrue
    assert p('Table[RandomReal[], {5}]  == Table[RandomReal[], {5}]') not in (True, BooleanTrue)


def test_subdivide():
    assert p('Subdivide[10]') == p('{0, 1/10, 1/5, 3/10, 2/5, 1/2, 3/5, 7/10, 4/5, 9/10, 1}')
    assert p('Subdivide[10, 5]') == p('{0, 2, 4, 6, 8, 10}')
    assert p('Subdivide[a, b, 6]') == p('{a, (5*a)/6 + b/6, (2*a)/3 + b/3, a/2 + b/2, a/3 + (2*b)/3, a/6 + (5*b)/6, b}')
    assert p('Subdivide[E, Pi, 4]') == p('{E, (3*E)/4 + Pi/4, E/2 + Pi/2, E/4 + (3*Pi)/4, Pi}')
    assert p('Subdivide[-1, 2, 5]') == p('{-1, -2/5, 1/5, 4/5, 7/5, 2}') == p('-1 + (2 - (-1)) Range[0, 5]/5')


def test_subsets():
    assert p('Subsets[{}]') == p('{{}}')
    assert p('Subsets[{a}]') == p('{{},{a}} ')
    assert p('Subsets[{a, b, c}]') == p('{{},{a},{b},{c},{a,b},{a,c},{b,c},{a,b,c}} ')
    assert p('Subsets[{a, b, c, d}, 2]') == p('{{},{a},{b},{c},{d},{a,b},{a,c},{a,d},{b,c},{b,d},{c,d}} ')
    assert p('Subsets[{a, b, c, d}, {2}]') == p('{{a,b},{a,c},{a,d},{b,c},{b,d},{c,d}}')
    assert p('Subsets[{a, b, c, d, e}, {3}, 5] ') == p('{{a,b,c},{a,b,d},{a,b,e},{a,c,d},{a,c,e}}')
    assert p('Subsets[{a, b, c, d, e}, {0, 5, 2}]') == \
           p("{{},{a,b},{a,c},{a,d},{a,e},{b,c},{b,d},{b,e},{c,d},{c,e},{d,e},"
             "{a,b,c,d},{a,b,c,e},{a,b,d,e},{a,c,d,e},{b,c,d,e}}")
    assert p('Subsets[{a, b, c, d}]') == \
           p("{{},{a},{b},{c},{d},{a,b},{a,c},{a,d},{b,c},{b,d},{c,d},{a,b,c},"
             "{a,b,d},{a,c,d},{b,c,d},{a,b,c,d}} ")
    assert p('Subsets[f[a, b, c]]') == p('{f[],f[a],f[b],f[c],f[a,b],f[a,c],f[b,c],f[a,b,c]}')
    assert p('Subsets[a + b + c]') == p('{0,a,b,c,a+b,a+c,b+c,a+b+c}')
    assert p('Subsets[{1, 2, 3, 4}, {3}]') == p('{{1,2,3},{1,2,4},{1,3,4},{2,3,4}}')
    assert p('Total[Subsets[Times[a, b, c, d, e], {3}]]') == \
           p("a*b*c + a*b*d + a*c*d + b*c*d + a*b*e + a*c*e + b*c*e + a*d*e + b*d*e + c*d*e")
    assert p('Subsets[Range[20], All, {69381}]') == p('{{1, 3, 4, 5, 11, 14, 17}}')
    assert p('Subsets[{a, b, c, d}, All, {15, 1, -2}]') == p('{{b,c,d},{a,b,d},{c,d},{b,c},{a,c},{d},{b},{}}')
    # assert p('Subsets[{a, b, b, b}]') == \
    #        p("{{},{a},{b},{b},{b},{a,b},{a,b},{a,b},{b,b},{b,b},{b,b},"
    #          "{a,b,b},{a,b,b},{a,b,b},{b,b,b},{a,b,b,b}}")


def test_length():
    assert p('Length[{a, b, c, d}]') == 4
    assert p('Length[a + b + c + d]') == 4
    assert p('Length[f[g[x, y], z]]') == 2
    assert p('Length[Sqrt[x]]') == 2


def test_first():
    assert p('First[{a, b, c}]') == p('a')
    assert p('First[{{a, b}, {c, d}}]') == p('{a, b}')
    assert p('First[a^2 + b^2]') == p('a^2')
    assert p('First[{}, x]') == p('x')
    assert p('First[{a, b}, x]') == p('a')
    assert p('First[1/a^2]') == p('a')


def test_last():
    assert p('Last[{a, b, c}]') == p('c')
    assert p('Last[{{a, b}, {c, d}}]') == p('{c, d}')
    assert p('Last[a^2 + b^2]') == p('b^2')
    assert p('Last[{}, x]') == p('x')
    assert p('Last[{a, b}, x]') == p('b')
    assert p('Last[1/a^2]') == p('-2')


def test_reverse():
    assert p('Reverse[{a, b, c, d}]') == p('{d,c,b,a}')
    assert p('Reverse[f[a, b, c]]') == p('f[c,b,a]')
    assert p('Reverse[Reverse[{a, b, c, d}]]') == p('{a,b,c,d}')
    assert p('Reverse[{{a,b,c},{d,e,f},{g,h,i}}]') == p('{{g,h,i},{d,e,f},{a,b,c}}')
    assert p('Reverse[{{a,b,c},{d,e,f},{g,h,i}},2]') == p('{{c,b,a},{f,e,d},{i,h,g}}')
    assert p('Reverse[{{a,b,c},{d,e,f},{g,h,i}},{1, 2}]') == p('{{i,h,g},{f,e,d},{c,b,a}}')
