from calc9000.functions.core import *
from calc9000.functions.base import Factorial
from calc9000.functions.list_funcs import Min, Max


class Round(NormalFunction):
    """
    Round [x]
     Gives the integer closest to x.

    Round [x, a]
     Rounds to the nearest multiple of a.
    """

    @classmethod
    def exec(cls, x, a=None):
        if x.is_number:
            if a is None:
                return round(x)
            return a * round(x / a)
        if isinstance(x, iterables):
            return thread(Round, x)
        return None


class Floor(NormalFunction):
    """
    Floor [x]
     Gives the greatest integer less than or equal to x.

    Floor [x,a]
     Gives the greatest multiple of a less than or equal to x.

    Uses sympy.floor().
    """

    @classmethod
    def exec(cls, x, a=None):
        if a is None:
            return thread(s.floor, x)
        return thread(lambda y: a * s.floor(y / a), x)


class Ceiling(NormalFunction):
    """
    Ceiling [x]
     Gives the smallest integer greater than or equal to x.

    Ceiling [x, a]
     Gives the smallest multiple of a greater than or equal to x.

    Uses sympy.ceiling().
    """

    @classmethod
    def exec(cls, x, a=None):
        if a is None:
            return thread(s.ceiling, x)
        return thread(lambda y: a * s.ceiling(y / a), x)


class Clip(NormalFunction):
    """
    Clip [x]
     Gives x clipped to be between and .

    Clip [x, {min, max}]
     Gives x for min≤x≤max, min for x<min and max for x>max.

    Clip [x, {min, max}, {v_min, v_max}]
     Gives v_min for x < min and v_max for x > max.
    """

    @classmethod
    def exec(cls, x, limits=(s.S.NegativeOne, s.S.One), limit_return=None):
        if isinstance(x, iterables):
            return List(*(Clip(i, limits, limit_return) for i in x))
        if limit_return is None:
            limit_return = limits
        if x.is_number:
            if not x.is_extended_real:
                raise FunctionException('Clip::com', 'Cannot clip complex values.')
            if x < limits[0]:
                return limit_return[0]
            if x > limits[1]:
                return limit_return[1]
            return x
            # return s.Max(s.Min(x, limits[1]), limits[0])
        return None

    def _eval_rewrite_as_Piecewise(self, **kwargs):
        if len(self.args) == 3:
            limits = self.args[1]
            limit_return = self.args[2]
        elif len(self.args) == 2:
            limits = self.args[1]
            limit_return = limits
        else:
            limits = (s.S.NegativeOne, s.S.One)
            limit_return = limits

        x = self.args[0]

        return s.Piecewise(
            (x, s.And(limits[0] <= x, x <= limits[1])),
            (limit_return[0], x < limits[0]),
            (limit_return[1], x > limits[1])
        )

    def _eval_derivative(self, x):
        return s.Derivative(self._eval_rewrite_as_Piecewise(), x, evaluate=True)

    def _eval_Integral(self, *args, **kwargs):
        return s.Integral(self._eval_rewrite_as_Piecewise(), *args)


class Rescale(NormalFunction):
    # TODO: clean

    tags = {
        'rescale': 'Invalid Arguments for Rescale.'
    }

    @classmethod
    def exec(cls, x, x_range=None, y_range=None):
        if x_range is None and isinstance(x, iterables) and y_range is None:
            x = list(x)
            _min = Min(x)
            _max = Max(x)
            for i in range(len(x)):
                x[i] = cls.eval(x[i], List(_min, _max))
            return List(*x)
        if isinstance(x_range, iterables) and len(x_range) == 2:
            if y_range is None or (isinstance(y_range, iterables) and len(y_range) == 2):
                if y_range is None:
                    y_range = (0, 1)
                return ((x - x_range[0]) * y_range[1] + (x_range[1] - x) * y_range[0]) / (x_range[1] - x_range[0])
        raise FunctionException('Rescale::rescale')  # TODO: ?


class Unitize(NormalFunction):
    """
    Unitize [x]
     Gives 0 when x is zero, and 1 when x has any other numerical value.
    """

    @staticmethod
    def _unitize(x):
        if s.ask(s.Q.zero(x)):
            return 0
        return 1

    @classmethod
    def exec(cls, x):
        return thread(cls._unitize, x)


class Ramp(NormalFunction):
    """
    Ramp [x]
     Gives x if x ≥ 0 and 0 otherwise.
    """

    @staticmethod
    def _ramp(x):
        if s.ask(s.Q.nonnegative(x)):
            return x
        return 0

    @classmethod
    def exec(cls, x):
        return thread(cls._ramp, x)


class Quotient(NormalFunction):
    @classmethod
    def exec(cls, m, n):
        if m.is_number and n.is_number:
            return m // n
        return None

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real


class QuotientRemainder(NormalFunction):
    """
    QuotientRemainder [m, n]
     Gives a list of the quotient and remainder from division of m by n.
    """

    @staticmethod
    def _qr(m, n):
        return List(m // n, m % n)

    @classmethod
    def exec(cls, m, n):
        thread(cls._qr, m, n)


class Rationalize(NormalFunction):
    """
    Rationalize [x]
     Converts an approximate number x to a nearby rational with small denominator.

    Rationalize [x, dx]
     Yields the rational number with smallest denominator that lies within dx of x.

    Rationalize [args, ForceRational -> False]
     Might return a better result that which may not necessarily be a rational.

    Uses sympy.nsimplify().
    See https://reference.wolfram.com/language/ref/Rationalize
    """

    @staticmethod
    def rat_rat(x, dx=None):
        pr = 20
        if dx:
            pr = -s.log(dx, 10) + 4
        rat = s.nsimplify(N(x, pr), tolerance=dx)
        if dx or 1 / (10 ** 4 * s.denom(rat) ** 2) > s.Abs(rat - x):
            if isinstance(rat, s.Rational):  # return better result
                return Rationalize.rat(x, dx)
            return rat
        return x

    @staticmethod
    def rat(x, dx=None):
        # TODO: add precision checking for 0, fix return
        pr = 20
        if dx:
            pr = -s.log(dx, 10) + 4
        rat = s.nsimplify(N(x, pr), rational=True, tolerance=dx)
        if dx or 1 / (10 ** 4 * s.denom(rat) ** 2) > s.N(s.Abs(rat - x)):
            return rat
        return x

    op_spec = ({'ForceRational': 'rat'}, {'rat': True})
    param_spec = (1, 2)

    @classmethod
    def exec(cls, x, dx=None, rat=True):
        if not rat:
            try:
                return thread(cls.rat_rat, x, dx)
            except AssertionError:  # this happens sometimes, dunno why
                return thread(cls.rat, x, dx)
        return thread(cls.rat, x, dx)


class nPr(NormalFunction):
    """
    nPr [n, r]
     Gives number of possibilities for choosing an ordered set of r objects from
     n objects.
    """

    @staticmethod
    def _npr(x, q):
        return Factorial(x) / Factorial(x - q)

    @classmethod
    def exec(cls, n, m):
        return thread(cls._npr, n, m)


class nCr(NormalFunction):
    """
    nCr [n, r]
     Gives The number of different, unordered combinations of r objects from n objects.
    """

    @staticmethod
    def _ncr(x, q):
        return Factorial(x) / (Factorial(x - q) * Factorial(q))

    @classmethod
    def exec(cls, n, m):
        return thread(cls._ncr, n, m)


class IntegerPart(NormalFunction):
    @staticmethod
    def integer_part(x):
        def non_complex_integer_part(n):
            if s.ask(s.Q.nonnegative(n)):
                return s.floor(n)
            return s.ceiling(n)

        if hasattr(x, "is_number") and x.is_number:
            return non_complex_integer_part(s.re(x)) + non_complex_integer_part(s.im(x)) * s.I

    @classmethod
    def exec(cls, x):
        return thread(cls.integer_part, x)


class FractionalPart(NormalFunction):
    @staticmethod
    def frac_part(x):
        if x.is_infinite:
            if x.is_complex or s.sign(x) > 0:
                return s.Interval(0, 1, True, True)
            return s.Interval(-1, 0, True, True)
        if hasattr(x, "is_number") and x.is_number:
            return x - IntegerPart.integer_part(x)
        return None

    @classmethod
    def exec(cls, x):
        return thread(cls.frac_part, x)
