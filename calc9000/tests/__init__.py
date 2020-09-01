import sympy as s
from calc9000.converse import process as p

f = s.Rational


def p_str(*args) -> str:
    return str(p(*args))


def p_int(*args) -> int:
    return int(p_str(*args))
