from calc9000.functions.core import *

# Trig Functions

Sinc = threaded('Sinc', s.sinc)
Sin = threaded('Sin', s.sin)
Cos = threaded('Cos', s.cos)
Tan = threaded('Tan', s.tan)
Csc = threaded('Csc', s.csc)
Sec = threaded('Sec', s.sec)
Cot = threaded('Cot', s.cot)
Sinh = threaded('Sinh', s.sinh)
Cosh = threaded('Cosh', s.cosh)
Tanh = threaded('Tanh', s.tanh)
Csch = threaded('Csch', s.csch)
Sech = threaded('Sech', s.sech)
Coth = threaded('Coth', s.coth)
ArcSin = threaded('ArcSin', s.asin)
ArcCos = threaded('ArcCos', s.acos)
ArcCsc = threaded('ArcCsc', s.acsc)
ArcSec = threaded('ArcSec', s.asec)
ArcCot = threaded('ArcCot', s.acot)
ArcSinh = threaded('ArcSinh', s.asinh)
ArcCosh = threaded('ArcCosh', s.acosh)
ArcTanh = threaded('ArcTanh', s.atanh)
ArcCsch = threaded('ArcCsch', s.acsch)
ArcSech = threaded('ArcSech', s.asech)
ArcCoth = threaded('ArcCoth', s.acoth)


class ArcTan(NormalFunction):
    @classmethod
    def exec(cls, y, x=None):
        if x is None:
            return thread(s.atan, y)
        return thread(s.atan2, x, y)


class Exp(NormalFunction):
    """
    Exp [z]
     Gives the exponential of z.
    """

    @classmethod
    def exec(cls, z):
        return thread(Power, s.E, z)


class Log(NormalFunction):
    """
    Log [z]
     Gives the natural logarithm of z (logarithm to base e).

    Log [b, z]
     Gives the logarithm to base b.
    """

    @classmethod
    def exec(cls, x, b=None):
        if b is not None:
            return thread(s.log, b, x)
        return thread(s.log, x)


class Log2(NormalFunction):
    """
    Log2 [z]
     Gives the base-2 logarithm of x.
    """

    @classmethod
    def exec(cls, x):
        return thread(s.log, x, 2)


class Log10(NormalFunction):
    """
    Log10 [z]
     Gives the base-10 logarithm of x.
    """

    @classmethod
    def exec(cls, x):
        return thread(s.log, x, 10)


class Re(NormalFunction):
    """
    Re [x]
     Gives the Real part of x.

    Equivalent to sympy.re().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.re, x)


class Im(NormalFunction):
    """
    Im [x]
     Gives the Imaginary part of x.

    Equivalent to sympy.im().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.im, x)


class ReIm(NormalFunction):
    """
    ReIm [x]
     Gives the list {Re[x], Im[x]} of x.
    """

    @classmethod
    def exec(cls, x):
        return thread(lambda b: List(Re(b), Im(b)), x)


class Plus(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Add, *args)


class Times(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Mul, *args)


class Power(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Pow, *args)


class Subtract(NormalFunction):
    @classmethod
    def exec(cls, x, y):
        return thread(s.Add, x, thread(s.Mul, -1, y))


class Divide(NormalFunction):
    @classmethod
    def exec(cls, x, y):
        return thread(s.Mul, x, thread(s.Pow, y, -1))


class Abs(NormalFunction):
    """
    Abs [x]
     Gives the absolute value of x.

    Equivalent to sympy.Abs().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.Abs, x)


class Arg(NormalFunction):
    """
    Arg [x]
     Gives the argument of the complex number x.

    Equivalent to sympy.arg().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.arg, x)


class AbsArg(NormalFunction):
    """
    AbsArg [z]
     Gives the list {Abs[z],Arg[z]} of the number z.
    """

    @classmethod
    def exec(cls, x):
        return thread(lambda y: List(s.Abs(y), s.arg(y)), x)


class Mod(NormalFunction):
    """
    Mod [m, n]
     Gives m mod n.

    Mod [m, n, d]
     Uses an offset d.

    Effectively uses sympy.Mod().
    """

    tags = {
        'complex': 'Mod does not support complex arguments.'
    }

    @classmethod
    def exec(cls, a, n, d=s.S.Zero):
        if any(isinstance(x, iterables) for x in (a, n, d)):
            return thread(Mod, a, n, d)

        if any(not x.is_extended_real for x in (a, n, d)):
            raise FunctionException('Mod::complex')

        if d:
            if a.is_number and n.is_number and d.is_number:
                return a - n * s.floor((a - d) / n)
            return None
        return s.Mod(a, n)

    def _eval_is_nonnegative(self):
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        if self.args[1].is_negative:
            return True

    def _eval_rewrite_as_floor(self, a, n, d, **kwargs):
        return a - n * s.floor((a - d) / n)


class Factorial(NormalFunction):
    """
    Factorial [x]
     Gives the Factorial of x.

    Equivalent to sympy.factorial().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.factorial, x)


class Conjugate(NormalFunction):
    """
    Conjugate [x]
     Gives the complex conjugate of complex number x.

    Equivalent to sympy.conjugate().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.conjugate, x)


class Sqrt(NormalFunction):
    """
    Sqrt [Expr]
     Gives the Square Root of Expr.

    Equivalent to sympy.sqrt().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.sqrt, x)


class Sign(NormalFunction):
    """
    Sign [x]
     Gives -1, 0, or 1 depending on whether x is negative, zero, or positive.

    For nonzero complex numbers.py z, Sign[z] is defined as z/Abs[z].
    """

    @staticmethod
    def _sign(n):
        if hasattr(n, 'is_real') and n.is_real:
            return s.sign(n)
        if n == s.oo:
            return 1
        if n == -s.oo:
            return -1
        return n / Abs(n)

    @classmethod
    def exec(cls, x):
        return thread(cls._sign, x)


class Equal(NormalFunction):
    """
    Equal [x1, x2, x3, …]
     Gives a condition x1 == x2 == x3 == …
    """

    @classmethod
    def exec(cls, *args):
        if len(args) == 1:
            return None
        if len(args) == 2:
            return s.Eq(args[0], args[1])
        # TODO: better multiple equality
        return s.And(*[s.Eq(args[x], args[x + 1]) for x in range(len(args) - 1)])
