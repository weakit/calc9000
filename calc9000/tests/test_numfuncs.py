from . import s, p, p_str, f, extra_precision
from calc9000.custom import List


def test_round():
    assert p('Round[2.4]') == 2
    assert p('Round[2.6]') == 3
    assert p('Round[226, 10]') == 230
    assert p('Round[-3.7]') == -4
    assert p_str('Round[-10.3, 3.5]').startswith('-10.50')
    assert p('Round[2 Pi - E, 5/4]') == f('15/4')
    assert p('Round[5.37 - 1.3 I]') == 5 - s.I
    assert p('Round[{2.4, 2.5, 2.6}]') == List(2, 2, 3)
    assert p('Round[0]') == 0
    assert p('Round[Infinity]') == s.oo


def test_floor():
    assert p('Floor[2.4]') == 2
    assert p('Floor[2.6]') == 2
    assert p('Floor[226, 10]') == 220
    assert p('Floor[5.37]') == 5
    assert p('Floor[-3.7]') == -4
    assert p_str('Floor[-10.3, 3.5]').startswith('-10.50')
    assert p('Floor[2 Pi - E, 5/4]') == f('5/2')
    assert p('Floor[5.37 - 1.3 I]') == 5 - 2 * s.I
    assert p('Floor[{2.4, 2.5, 2.6}]') == List(2, 2, 2)
    assert p('Floor[0]') == 0
    assert p('Floor[Infinity]') == s.oo
    assert p('Simplify[Floor[x + 1]]') == p('1 + Floor[x]')


def test_ceiling():
    assert p('Ceiling[2.4]') == 3
    assert p('Ceiling[2.6]') == 3
    assert p('Ceiling[226, 10]') == 230
    assert p('Ceiling[5.37]') == 6
    assert p('Ceiling[-3.7]') == -3
    assert p_str('Ceiling[-10.3, 3.5]').startswith('-7.0')
    assert p('Ceiling[2 Pi - E, 5/4]') == f('15/4')
    assert p('Ceiling[5.37 - 1.3 I]') == 6 - s.I
    assert p('Ceiling[{2.4, 2.5, 2.6}]') == List(3, 3, 3)
    assert p('Ceiling[0]') == 0
    assert p('Ceiling[Infinity]') == s.oo
    assert p('Simplify[Ceiling[x + 1]]') == p('1 + Ceiling[x]')


def test_int_part():
    assert p('IntegerPart[2.4]') == 2
    assert p('IntegerPart[-2.4]') == -2
    assert p('IntegerPart[456.457]') == 456
    assert p('IntegerPart[-5/4]') == -1
    assert p('IntegerPart[Pi + E]') == 5
    assert p('IntegerPart[78/47 - 4.2 I]') == 1 - 4 * s.I
    assert p('IntegerPart[{2.4, 2.5, 3.0}]') == List(2, 2, 3)
    assert p('IntegerPart[0]') == 0
    assert p('IntegerPart[Infinity]') == s.oo


def test_frac_part():
    assert p_str('FractionalPart[2.4]').startswith('0.4')
    assert p_str('FractionalPart[-2.4]').startswith('-0.4')
    assert p_str('N[FractionalPart[456.457], 3]').startswith('0.457')
    assert p('FractionalPart[-5/4]') == f('-1/4')
    assert p_str('N[FractionalPart[Pi + E]]').startswith('0.859')
    assert s.re(p('FractionalPart[235/47 + 5.3 I]')) == 0 and \
           str(s.im(p('FractionalPart[235/47 + 5.3 I]'))).startswith('0.3')
    assert p('FractionalPart[0]') == 0


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


def test_abs():
    assert p_str('Abs[-2.5]').startswith('2.50')
    assert p_str('Abs[3.14]').startswith('3.140')
    assert p('Abs[3 + 4 I]') == 5
    assert p_str('Abs[1.4 + 2.3 I]').startswith('2.69258')
    assert p('Abs[0]') == 0
    assert p('Abs[Infinity]') == s.oo
    assert p('Abs[I Infinity]') == s.oo


def test_sign():
    assert p('Sign[-2.5]') == -1
    assert p('Sign[3.14]') == 1
    assert p('Sign[1 + I]') == (1 + s.I)/s.sqrt(2)
    assert p('Sign[{1.2, 1.5, -1.8}]') == List(1, 1, -1)
    assert p('Sign[0]') == 0
    assert p('Sign[Infinity]') == 1
    assert p('Sign[-Infinity]') == -1


def test_clip():
    assert p('Clip[8.5]') == 1
    assert p('Clip[-5/2, {-2, 2}]') == -2
    assert p('Clip[Pi, {-9, 7}, {11, 28}]') == s.pi
    assert p('Clip[3 Pi, {-9, 7}, {11, 28}]') == 28
    assert p('Clip[{-2, 0, 2}]') == List(-1, 0, 1)
    assert p('Clip[0]') == 0
    assert p('Clip[Infinity]') == 1


def test_rescale():
    assert p_str('Rescale[2.5, {-10, 10}]').startswith('0.6250')
    assert p_str('Rescale[12.5, {-10, 10}]').startswith('1.1250')
    assert p('Rescale[-3/2, {-2, 2}]') == f(1, 8)
    assert p_str('N[Rescale[Pi, {0, 2.5}]]').startswith('1.2566')
    assert p('Rescale[3, {-9, 7}, {11, 28}]') == f(95, 4)
    assert p('Expand[Rescale[1 + 2 I, {0, 5}]]') == f(1, 5) + 2 * s.I / 5
    assert p('Expand[Rescale[1 + 2 I, {0, 1 + I}]]') == f(3, 2) + s.I / 2
    assert p('Rescale[{-2, 0, 2}]') == List(0, f(1, 2), 1)
    assert p('Rescale[{-2, 0, 2}, {-5, 5}]') == List(f(3, 10), f(1, 2), f(7, 10))
    assert p('Rescale[{-2, 0, 2}, {-5, 5}, {-1, 1}]') == List(f(-2, 5), 0, f(2, 5))


def test_unitize():
    assert p('Unitize[Pi]') == 1
    assert p('Unitize[Sqrt[2] + Sqrt[3] - Sqrt[5 + 2 Sqrt[6]]]') == 0
    assert p('m = {{-2, 1, 0, 1}, {1, -2, 1, 0}, {0, 1, -2, 1}, {1, 0, -2, 1}}; Unitize[m]') == \
           List(List(1, 1, 0, 1), List(1, 1, 1, 0), List(0, 1, 1, 1), List(1, 0, 1, 1))


def test_ramp():
    assert p('Ramp[-1]') == 0
    assert p_str('Ramp[3.7]').startswith('3.70')
    assert p('Ramp[1/2]') == f(1, 2)
    assert p('Ramp[-E]') == 0
    assert p('Ramp[-1.55]') == 0
    assert p('Ramp[{-5, -1.5, 0, 1/3, Pi}]') == List(0, 0, 0, f(1, 3), s.pi)
