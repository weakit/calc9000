from calc9000.functions.base import ArcTan, Sqrt, Times
from calc9000.functions.core import *
from calc9000.functions.list_funcs import Total


class DiracDelta(NormalFunction):
    """
    DiracDelta [x]
     represents the Dirac delta function δ(x).

    DiracDelta [x1, x2, …]
     Represents the multidimensional Dirac delta function δ(x1, x2, …).

    Equivalent to sympy.DiracDelta().
    """

    @classmethod
    def exec(cls, *args):
        return Times(*thread(s.DiracDelta, args))


class HeavisideTheta(NormalFunction):
    """
    HeavisideTheta [x]
     Represents the Heaviside theta function θ(x), equal to 0 for x < 0 and 1 for x > 0.

    Equivalent to sympy.Heaviside().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.Heaviside, x)


class Gamma(NormalFunction):
    @classmethod
    def exec(cls, x):
        # TODO: incomplete gamma
        return thread(s.gamma, x)


class PolyGamma(NormalFunction):
    @classmethod
    def exec(cls, n, z=None):
        if z is None:
            return Functions.call("PolyGamma", 0, n)
        return thread(s.polygamma, n, z)


class StieltjesGamma(NormalFunction):
    @classmethod
    def exec(cls, x, a=None):
        if a:
            return thread(s.stieltjes, x, a)
        return thread(s.stieltjes, x)


class LogisticSigmoid(NormalFunction):  # why is this here?
    """
    LogisticSigmoid [z]
     Gives the logistic sigmoid function.
    """

    @classmethod
    def exec(cls, z):
        return thread(lambda x: 1 / (1 + s.exp(-x)), z)


class Sum(NormalFunction):

    tags = {"sum": "Invalid Limits."}

    @classmethod
    def limits(cls, a, b):
        if isinstance(a, s.Symbol) or (isinstance(b, s.Symbol) and is_integer(a)):
            return a, b
        return a, b - s.Mod(b - a, 1)

    @classmethod
    def process_skip(cls, s_, i):
        return s_.subs(i[0], i[0] * i[3]), (i[0], *cls.limits(i[1] / i[3], i[2] / i[3]))

    @classmethod
    def process(cls, s_, i):
        if isinstance(i, iterables):
            if not isinstance(i[0], s.Symbol):
                raise FunctionException("Sum::sum")
            if len(i) == 2:
                return s_, (i[0], *cls.limits(s.S.One, i[1]))
            if len(i) == 3:
                return s_, (i[0], *cls.limits(i[1], i[2]))
            if len(i) == 4:
                return cls.process_skip(s_, i)
        raise FunctionException("Sum::sum")

    @classmethod
    def exec(cls, f, i, *xi):
        i = (i,) + xi
        sum_ = f
        for limit in reversed(i):
            if isinstance(limit, iterables):
                sum_ = thread(lambda s_: s.summation(*cls.process(s_, limit)), sum_)
            else:
                # sum_ = thread(s.concrete.gosper.gosper_sum, sum_, limit)
                raise NotImplementedError  # raze hell
        return sum_


class Product(Sum):
    @classmethod
    def exec(cls, f, i, *xi):
        i = (i,) + xi
        product = f
        for limit in reversed(i):
            if isinstance(limit, iterables):
                product = s.product(*cls.process(product, limit))
            else:
                raise NotImplementedError  # be nicer
        return product


class Zeta(NormalFunction):
    """
    Zeta [s]
     Gives the Riemann zeta function ζ(s).

    Zeta [s, a]
     Gives the Generalized Riemann zeta function ζ(s, a).

    Equivalent to sympy.zeta().
    """

    @classmethod
    def exec(cls, n, a=1):
        return thread(s.zeta, n, a)


class FromPolarCoordinates(NormalFunction):
    """
    FromPolarCoordinates [{r, θ}]
     Gives the {x, y} Cartesian coordinates corresponding to the
     polar coordinates {r, θ}.

    FromPolarCoordinates [{r, θ1, …, θn - 2, ϕ}]
     Gives the coordinates corresponding to the hyperspherical
     coordinates {r, θ1, …, θn - 2, ϕ}
    """

    tags = {
        "dim": "Polar Coordinates can only be defined for dimensions of 2 and greater."
    }

    @classmethod
    def exec(cls, list_):
        # TODO: Thread
        length = len(list_)
        if length == 1:
            raise FunctionException("FromPolarCoordinates::dim")
        ret = List.create(list_[:1] * length)
        for pos, angle in enumerate(list_[1:]):
            ret[pos] *= s.cos(angle)
            for x in range(pos + 1, length):
                ret[x] *= s.sin(angle)
        return ret


class ToPolarCoordinates(NormalFunction):
    """
    ToPolarCoordinates [{x, y}]
     Gives the {r, θ} polar coordinates corresponding to the Cartesian
     coordinates {x, y}.

    ToPolarCoordinates [{x1, x2, …, xn}]
     Gives the hyperspherical coordinates corresponding to the Cartesian
     coordinates {x1, x2, …, xn}.
    """

    tags = {
        "dim": "Polar Coordinates can only be defined for dimensions of 2 and greater."
    }

    @classmethod
    def exec(cls, list_):
        # TODO: Thread, don't use append
        list_ = List(*list_)
        length = len(list_)
        if length == 1:
            raise FunctionException("ToPolarCoordinates::dim")
        ret = List(Sqrt(Total(list_ ** 2)))
        for x in range(length - 2):
            ret.append(s.acos(list_[x] / Sqrt(Total(list_[x:] ** 2))))
        ret.append(ArcTan(list_[-2], list_[-1]))
        return List(*ret)


class Timing(ExplicitFunction):
    @classmethod
    def exec(cls, expr):
        import time

        start = time.time()
        result = LazyFunction.evaluate(expr)
        end = time.time()
        return List(
            s.Float(end) - s.Float(start),
            result if not isinstance(result, NoOutput) else r.Constants.Null,
        )


class Normal(NormalFunction):
    @staticmethod
    def do_normal(expr):
        if expr.has(s.Order):
            expr = expr.removeO()

        return expr

    @classmethod
    def exec(cls, expr):
        return thread(cls.do_normal, expr)
