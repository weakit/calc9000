from . import s, p, p_int, p_str
from calc9000.datatypes import List


def test_round():
    assert p_int('Round[2.4]') == 2
    assert p_int('Round[2.6]') == 3
    assert p_int('Round[226, 10]') == 230
    assert p_int('Round[-3.7]') == -4
    assert p_str('Round[-10.3, 3.5]').startswith('-10.50')
    assert p('Round[2 Pi - E, 5/4]') == s.Rational('15/4')
    assert p('Round[5.37 - 1.3 I]') == 5 - s.I
    assert p('Round[{2.4, 2.5, 2.6}]') == List(2, 2, 3)
    assert p_int('Round[0]') == 0
    assert p('Round[Infinity]') == s.oo


def test_floor():
    assert p_int('Floor[2.4]') == 2
    assert p_int('Floor[2.6]') == 2
    assert p_int('Floor[226, 10]') == 220
    assert p_int('Floor[5.37]') == 5
    assert p_int('Floor[-3.7]') == -4
    assert p_str('Floor[-10.3, 3.5]').startswith('-10.50')
    assert p('Floor[2 Pi - E, 5/4]') == s.Rational('5/2')
    assert p('Floor[5.37 - 1.3 I]') == 5 - 2 * s.I
    assert p('Floor[{2.4, 2.5, 2.6}]') == List(2, 2, 2)
    assert p_int('Floor[0]') == 0
    assert p('Floor[Infinity]') == s.oo
    assert p('Simplify[Floor[x + 1]]') == p('1 + Floor[x]')


def test_ceiling():
    assert p_int('Ceiling[2.4]') == 3
    assert p_int('Ceiling[2.6]') == 3
    assert p_int('Ceiling[226, 10]') == 230
    assert p_int('Ceiling[5.37]') == 6
    assert p_int('Ceiling[-3.7]') == -3
    assert p_str('Ceiling[-10.3, 3.5]').startswith('-7.0')
    assert p('Ceiling[2 Pi - E, 5/4]') == s.Rational('15/4')
    assert p('Ceiling[5.37 - 1.3 I]') == 6 - s.I
    assert p('Ceiling[{2.4, 2.5, 2.6}]') == List(3, 3, 3)
    assert p_int('Ceiling[0]') == 0
    assert p('Ceiling[Infinity]') == s.oo
    assert p('Simplify[Ceiling[x + 1]]') == p('1 + Ceiling[x]')


def test_int_part():
    assert p_int('IntegerPart[2.4]') == 2
    assert p_int('IntegerPart[-2.4]') == -2
    assert p_int('IntegerPart[456.457]') == 456
    assert p_int('IntegerPart[-5/4]') == -1
    assert p_int('IntegerPart[Pi + E]') == 5
    assert p('IntegerPart[78/47 - 4.2 I]') == 1 - 4 * s.I
    assert p('IntegerPart[{2.4, 2.5, 3.0}]') == List(2, 2, 3)
    assert p_int('IntegerPart[0]') == 0
    assert p('IntegerPart[Infinity]') == s.oo


def test_frac_part():
    assert p_str('FractionalPart[2.4]').startswith('0.4')
    assert p_str('FractionalPart[-2.4]').startswith('-0.4')
    assert p_str('N[FractionalPart[456.457]]').startswith('0.457')
    assert p('FractionalPart[-5/4]') == s.Rational('-1/4')
    assert p_str('N[FractionalPart[Pi + E]]').startswith('0.859')
    assert s.re(p('FractionalPart[235/47 + 5.3 I]')) == 0 and \
           str(s.im(p('FractionalPart[235/47 + 5.3 I]'))).startswith('0.3')
    assert p_int('FractionalPart[0]') == 0
