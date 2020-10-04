from . import s, p, p_str, extra_precision


def test_constants():
    assert p_str(f'N[Pi, 15-{extra_precision}]') == '3.14159265358979'
    assert p_str(f'N[E, 15-{extra_precision}]') == '2.71828182845905'
    assert p_str(f'N[EulerGamma, 15-{extra_precision}]') == '0.577215664901533'
    assert p_str(f'N[GoldenRatio, 15-{extra_precision}]') == '1.61803398874989'
    assert p_str(f'N[Catalan, 15-{extra_precision}]') == '0.915965594177219'
    assert p_str(f'N[Degree, 15-{extra_precision}]') == '0.0174532925199433'


def test_operations():
    assert p('1 + 1') == 2
    assert p('1 + 2 + 3') == 6
    assert p_str('5 + 2*3 - 7.5').startswith('3.50')
    assert p('((5 - 3)^(1 + 2))/4') == 2
    assert p('1/4 + 1/3') == s.S('7/12')
    assert p_str('100!') == '933262154439441526816992388562667004907159682643816214685929638952175999932299' \
        '15608941463976156518286253697920827223758251185210916864000000000000000000000000'
    assert p_str(f'N[100!, 5-{extra_precision}]') == '9.3326e+157'
    assert p_str('.25 + 1/3').startswith('0.5833')
    assert p('2^3') == 8
    assert p('2 3') == 6
    assert p_str('Sqrt[3.]').startswith('1.732')
    assert p('Sqrt[-25]') == 5 * s.I
    assert p('Subtract[10, 3]') == 7
    assert p('Plus[1, 2]') == 3
    assert p_str('616/33') == '56/3'

    a, b, c = s.var('a b c')
    assert p('a / b / c') == a / (b*c)
    assert p('a + a') == 2 * a
    assert p('a - a') == 0
