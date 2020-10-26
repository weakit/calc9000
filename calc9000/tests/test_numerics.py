import sympy as s

from . import f, p, p_str


def test_rationalize():
    assert p("Rationalize[6.75]") == f(27, 4)
    assert p_str("N[Rationalize[Pi, .01], 10]").startswith("3.14")
    assert p_str("N[Rationalize[Pi, .0001], 10]").startswith("3.1415")
    assert p_str("N[Rationalize[Exp[Sqrt[2]], 10^-2]]").startswith("4.11")
    assert p("Rationalize[1.2 + 6.7 x]") == f(6, 5) + f(67, 10) * s.Symbol("x")
