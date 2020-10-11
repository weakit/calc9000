from calc9000.functions.core import *
from calc9000.functions.base import Re, Im

import random as py_random
from mpmath.libmp import from_man_exp, dps_to_prec


randint = py_random.randint
get_rand_bits = py_random.getrandbits


def rand_float(p):
    return s.Float._new(from_man_exp(get_rand_bits(p), -p), p)


class RandomInteger(NormalFunction):
    """
    RandomInteger []
     Pseudo-randomly gives 0 or 1.

    RandomInteger [i]
     Gives a pseudo-random integer in the range {0, ..., i}.

    RandomInteger [{min, max}]
     Gives a pseudo-random integer in the range {min, max}.

    RandomInteger [range, n]
     Gives a list of n pseudo-random integers.

    RandomInteger [range, {n1, n2, …}]
     Gives an n1 × n2 × … array of pseudo-random integers.

    Effectively uses secrets.SystemRandom().
    """

    @classmethod
    def exec(cls, spec=None, rep=None):
        if not spec:
            return s.Integer(randint(0, 1))
        if not rep:
            if isinstance(spec, iterables):
                if len(spec) != 2:
                    raise FunctionException('RandomInteger::bounds', 'Invalid Bounds')
                limit = spec
            else:
                limit = [0, spec]
            if not (is_integer(limit[0]) and is_integer(limit[1])):
                raise FunctionException('RandomInteger::limits', 'Limits for RandomInteger should be an Integer.')
            return s.Integer(randint(limit[0], limit[1]))
        return cls.repeat(spec, rep, RandomInteger)

    @classmethod
    def repeat(cls, spec, rep, func):
        if is_integer(rep):
            return List(*[func.exec(spec) for _ in range(int(rep))])
        if isinstance(rep, iterables):
            if len(rep) == 1:
                return func.exec(spec, rep[0])
            return List(*[func.exec(spec, rep[1:]) for _ in range(int(rep[0]))])
        raise FunctionException('RandomInteger::bounds', "Invalid Bounds")


class RandomReal(NormalFunction):
    """
    RandomReal []
     Gives a pseudo-random real number in the range 0 to 1

    RandomReal [i]
     Gives a pseudo-random real number in the range 0 to i.

    RandomReal [{min, max}]
     Gives a pseudo-random real number in the range min to max.

    RandomReal [range, n]
     Gives a list of n pseudo-random reals.

    RandomReal [range, {n1, n2, …}]
     Gives an n1 × n2 × … array of pseudo-random reals.

    Effectively uses secrets.SystemRandom().
    """

    op_spec = ({'WorkingPrecision': 'p'}, {'p': DefaultPrecision})
    param_spec = (0, 2)

    @classmethod
    def exec(cls, spec=None, rep=None, p=DefaultPrecision):
        precision = p + ExtraPrecision
        mpmath_precision = dps_to_prec(precision)
        return cls.random(spec, rep, mpmath_precision)

    @classmethod
    def random(cls, spec=None, rep=None, p=53):
        if spec is None:
            return rand_float(p)
        if not rep:
            if isinstance(spec, iterables):
                if len(spec) != 2:
                    raise FunctionException('RandomReal::bounds', 'Invalid Bounds')
                lower, upper = spec
            else:
                lower, upper = 0, spec
            return lower + ((upper - lower) * rand_float(p))
        return cls.repeat(spec, rep, p=p)

    @classmethod
    def repeat(cls, spec, rep, p=DefaultPrecision):
        precision = p
        if is_integer(rep):
            return List(*[cls.random(spec, p=precision) for _ in range(int(rep))])
        if isinstance(rep, iterables):
            if len(rep) == 1:
                return cls.random(spec, rep[0], p=precision)
            return List(*[cls.random(spec, rep[1:], p=precision) for _ in range(int(rep[0]))])
        raise FunctionException('RandomReal::bounds', "Invalid Bounds")


class RandomComplex(NormalFunction):
    """
    RandomComplex []
     Gives a pseudo-random complex number with real and
     imaginary parts in the range 0 to 1.

    RandomComplex [{min, max}]
     Gives a pseudo-random complex number in the rectangle with
     corners given by the complex numbers.py min and max.

    RandomComplex [max]
     Gives a pseudo-random complex number in the rectangle whose
     corners are the origin and max.

    RandomComplex [range, n]
     Gives a list of n pseudo-random complex numbers.py.

    RandomComplex [range, {n1, n2, …}]
     Gives an n1 × n2 × … array of pseudo-random complex numbers.py.

    Effectively uses secrets.SystemRandom().
    """

    op_spec = ({'WorkingPrecision': 'p'}, {'p': DefaultPrecision})
    param_spec = (0, 2)

    @classmethod
    def exec(cls, spec=None, rep=None, p=DefaultPrecision):
        if spec:
            if rep:
                return RandomReal.exec(Re(spec), rep, p) + RandomReal.exec(Im(spec), rep, p) * s.I
            return RandomReal.exec(Re(spec), p=p) + RandomReal.exec(Im(spec), p=p) * s.I
        return RandomReal() + RandomReal() * s.I


class RandomPrime(NormalFunction):
    """
    RandomPrime [{a, b}]
     Gives a pseudorandom prime number in the range a to b.

    RandomPrime [x]
     Gives a pseudorandom prime number in the range 2 to x.

    RandomPrime [range, n]
     Gives a list of n pseudorandom primes.

    Effectively uses sympy.randprime().
    """

    tags = {
        'int': 'A positive integer is expected as input.'
    }

    @classmethod
    def exec(cls, spec, n=None):
        if not isinstance(spec, iterables):
            spec = List(2, spec)
        if not is_integer(spec[0]) and not is_integer(spec[1]):
            FunctionException(f'RandomPrime::int')

        if n:
            if is_integer(n):
                return List(*(cls.exec(spec) for _ in range(n)))
            if isinstance(n, iterables):
                return List(*(cls.exec(spec, n[1:]) for _ in range(n[0])))
        return s.randprime(spec[0], spec[1])
