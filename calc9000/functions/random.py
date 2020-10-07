from calc9000.functions.core import *
from calc9000.functions.base import Re, Im

import secrets
random = secrets.SystemRandom()


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
            return s.Rational(random.randint(0, 1))
        if not rep:
            if isinstance(spec, iterables):
                if len(spec) != 2:
                    raise FunctionException('RandomInteger::bounds', 'Invalid Bounds')
                limit = spec
            else:
                limit = [0, spec]
            if not (is_integer(limit[0]) and is_integer(limit[1])):
                raise FunctionException('RandomInteger::limits', 'Limits for RandomInteger should be an Integer.')
            return s.Rational(random.randint(limit[0], limit[1]))
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

        if not spec:
            return random.randint(0, 10 ** precision) * s.N(10 ** (-precision), precision)
        if not rep:
            if isinstance(spec, iterables):
                if len(spec) != 2:
                    raise FunctionException('RandomReal::bounds', 'Invalid Bounds')
                lower, upper = spec
            else:
                lower, upper = 0, spec
            return lower + random.randint(0, (upper - lower) * 10 ** precision) * s.N(10 ** (-precision), precision)
        return cls.repeat(spec, rep, p=precision)

    @classmethod
    def repeat(cls, spec, rep, p=DefaultPrecision):
        precision = p
        if is_integer(rep):
            return List(*[cls.exec(spec, p=precision) for _ in range(int(rep))])
        if isinstance(rep, iterables):
            if len(rep) == 1:
                return cls.exec(spec, rep[0], p=precision)
            return List(*[cls.exec(spec, rep[1:], p=precision) for _ in range(int(rep[0]))])
        raise FunctionException('RandomReal::bounds', "Invalid Bounds")


class RandomComplex(RandomReal):
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
        precision = p

        if not spec:
            return RandomReal.exec(p=precision) + RandomReal.exec(p=precision) * r.refs.Constants.I
        if not rep:
            return RandomReal.exec(Re.exec(spec), p=precision) + \
                   RandomReal.exec(Im.exec(spec), p=precision) * r.refs.Constants.I
        return cls.repeat(spec, rep, p=precision)
