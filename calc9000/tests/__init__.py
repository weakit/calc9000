import sympy as s
from calc9000.converse import process as p
from calc9000.references import refs

f = s.Rational
extra_p = refs.ExtraPrecision


def p_str(input_str: str) -> str:
    return str(p(input_str))


def p_int(input_str: str) -> int:
    return int(p_str(input_str))
