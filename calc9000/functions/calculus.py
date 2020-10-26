from calc9000.functions.core import *


class D(NormalFunction):
    """
    D [f, x]
     Gives the partial derivative ∂f / ∂x.

    D [f, {x, n}]
     Gives the multiple derivative ∂^n f / ∂ x^n.

    D [f, x, y, …]
     Gives the partial derivative (∂ / ∂y) (∂ / ∂x) f.

    D [f, {x, n}, {y, m}, …]
     Gives the multiple partial derivative (∂^m / ∂ y^m) (∂^n / ∂ x^n) f.

    D [f, {{x1, x2, …}}]
     For a scalar f gives the vector derivative (∂f / ∂x1, ∂f / ∂x2, …).

    Effectively uses sympy.diff().
    """

    tags = {
        "argx": "No variable of differentiation was specified to differentiate a multi-variate expression.",
        "spec": "Invalid specification for D.",
    }

    @classmethod
    def exec(cls, f, *args):
        def threaded_diff(x, *d):
            if isinstance(x, iterables):
                return List.create(threaded_diff(element, *d) for element in x)
            return s.diff(x, *d)

        if not args:
            try:
                return s.diff(f)
            except ValueError:
                raise FunctionException("D::argx")

        for arg in args:
            if isinstance(arg, iterables):
                if len(arg) == 1 and isinstance(arg[0], iterables):
                    return List.create(threaded_diff(f, element) for element in arg[0])
                if len(arg) == 2:
                    if isinstance(arg[0], iterables):
                        f = List.create(
                            threaded_diff(f, (element, arg[1])) for element in arg[0]
                        )
                    else:
                        f = threaded_diff(f, (arg[0], arg[1]))
                else:
                    raise FunctionException("D::spec")
            else:
                f = threaded_diff(f, arg)
        return f


class Integrate(NormalFunction):
    # TODO: Doc
    # TODO: return None if cannot find integral

    tags = {
        "argx": "No variable of integration was specified to integrate a multi-variate expression."
    }

    @classmethod
    def exec(cls, f, *args):
        def threaded_int(x, *i):
            if isinstance(x, iterables):
                return List.create(threaded_int(element, *i) for element in x)
            return s.integrate(x, *i)

        if not args:
            try:
                return s.integrate(f)
            except ValueError:
                raise FunctionException("Integrate::argx")

        return threaded_int(f, *args)


class Limit(NormalFunction):
    tags = {"lim": "Invalid Limit.", "dir": "Invalid Limit Direction."}

    @staticmethod
    def lim(expr, lim, d="+-"):
        if not isinstance(lim, Rule):
            raise FunctionException("Limit::lim")
        try:
            return s.limit(expr, lim.lhs, lim.rhs, d)
        except ValueError as e:
            if e.args[0].startswith("The limit does not exist"):
                return s.nan

    op_spec = ({"Direction": "d"}, {"d": "+-"})
    param_spec = (2, 2)
    rule_param = True

    @classmethod
    def exec(cls, expr, lim, d="+-"):
        if isinstance(d, String):
            if d.value in ("Reals", "TwoSided"):
                d = "+-"
            elif d.value in ("FromAbove", "Right") or d == -1:
                d = "+"
            elif d.value in ("FromBelow", "Left") or d == 1:
                d = "-"
        elif is_integer(d):
            if d == -1:
                d = "+"
            elif d == 1:
                d = "-"
        if d not in ("+", "-", "+-"):
            raise FunctionException("Limit::dir")
        return thread(lambda x: Limit.lim(x, lim, d), expr)


class LogIntegral(NormalFunction):
    """
    LogIntegral [z]
     gives the logarithmic integral function li(z).

    Equivalent to sympy.li().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.li, z)


class ExpIntegralEi(NormalFunction):
    """
    ExpIntegralEi [z]
     Gives the exponential integral function Ei(z).

    Equivalent to sympy.Ei().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.Ei, z)


class ExpIntegralE(NormalFunction):
    """
    ExpIntegralE [n, z]
     Gives the exponential integral function En(z).

    Equivalent to sympy.expint()
    """

    @classmethod
    def exec(cls, n, z):
        return thread(s.expint, n, z)


class SinIntegral(NormalFunction):
    """
    SinIntegral [z]
     Gives the sine integral function Si(z).

    Equivalent to sympy.Si().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.Si, z)


class CosIntegral(NormalFunction):
    """
    CosIntegral [z]
     Gives the cosine integral function Ci(z).

    Equivalent to sympy.Ci().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.Ci, z)


class SinhIntegral(NormalFunction):
    """
    SinhIntegral [z]
     Gives the hyperbolic sine integral function Si(z).

    Equivalent to sympy.Shi().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.Shi, z)


class CoshIntegral(NormalFunction):
    """
    CoshIntegral [z]
     Gives the hyperbolic cosine integral function Chi(z).

    Equivalent to sympy.Chi().
    """

    @classmethod
    def exec(cls, z):
        return thread(s.Chi, z)


class EllipticK(NormalFunction):
    """
    EllipticK [m]
     Gives the complete elliptic integral of the first kind

     .. math:: K(m) = F\left(\tfrac{\pi}{2}\middle| m\right)

     where $F\left(z\middle| m\right)$ is the Legendre incomplete
     elliptic integral of the first kind.

    Equivalent to sympy.elliptic_k().
    """

    @classmethod
    def exec(cls, m):
        return thread(s.elliptic_k, m)


class EllipticF(NormalFunction):
    r"""
    EllipticF [m]
     Gives the Legendre incomplete elliptic integral of the first kind

     .. math:: F\left(z\middle| m\right) =
              \int_0^z \frac{dt}{\sqrt{1 - m \sin^2 t}}

    Equivalent to sympy.elliptic_f().
    """

    @classmethod
    def exec(cls, m):
        return thread(s.elliptic_f, m)


class EllipticE(NormalFunction):
    r"""
    EllipticE [m]
     Gives the Legendre complete elliptic integral of the second kind

     .. math:: E(m) = E\left(\tfrac{\pi}{2}\middle| m\right)

    EllipticE [z, m]
     Gives the incomplete elliptic integral of the second kind

     .. math:: E\left(z\middle| m\right) = \int_0^z \sqrt{1 - m \sin^2 t} dt

    Equivalent to sympy.elliptic_e().
    """

    @classmethod
    def exec(cls, a, b=None):
        if b is not None:
            return thread(s.elliptic_e, a, b)
        return thread(s.elliptic_e, a)


class EllipticPi(NormalFunction):
    r"""
    EllipticPi [n, m]
     Gives the complete elliptic integral of the third kind

     .. math:: \Pi\left(n\middle| m\right) =
              \Pi\left(n; \tfrac{\pi}{2}\middle| m\right)

    EllipticPi [n, z, m]
     Gives the Legendre incomplete elliptic integral of the third kind

     .. math:: \Pi\left(n; z\middle| m\right) = \int_0^z \frac{dt}
              {\left(1 - n \sin^2 t\right) \sqrt{1 - m \sin^2 t}}

    Equivalent to sympy.elliptic_pi().
    """

    @classmethod
    def exec(cls, a, b, c=None):
        if c is not None:
            return thread(s.elliptic_pi, a, b, c)
        return thread(s.elliptic_pi, a, b)


class Erf(NormalFunction):
    """
    Erf [z]
     Gives the gauss error function erf(z).

    Erf[x, y]
     Gives the generalized error function erf(x) - erf(y).

    Equivalent to sympy.erf() and sympy.erf2().
    """

    @classmethod
    def exec(cls, x, y=None):
        if y is not None:  # TODO: fix over at sympy
            return thread(s.erf2, x, y)
        return thread(s.erf, x)


class Erfc(NormalFunction):
    """
    Erfc [z]
     Gives the complementary error function erfc(z).

    Equivalent to sympy.erfc().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.erfc, x)


class Erfi(NormalFunction):
    """
    Erfi [z]
     Gives the imaginary error function erfi(z).

    Equivalent to sympy.erfi().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.erfi, x)


class InverseErf(NormalFunction):
    """
    InverseErf [z]
     Gives the inverse gauss error function function erfc(z).

    Equivalent to sympy.erfinv().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.erfinv, x)


class InverseErfc(NormalFunction):
    """
    InverseErfc [z]
     Gives the inverse complementary error function function erfc(z).

    Equivalent to sympy.erfcinv().
    """

    @classmethod
    def exec(cls, x):
        return thread(s.erfcinv, x)


FresnelS = threaded("FresnelS", s.fresnels)
FresnelC = threaded("FresnelC", s.fresnelc)


class Series(NormalFunction):
    """
    Series[f, x -> x0]
     Gives the series expansion of f around point x = x0.

    Series[f, {x, x0, n}]
     Gives the series expansion of f around point x = x0, with n terms.

    Equivalent to sympy.series().
    """

    tags = {
        'specx': 'Invalid series specification given.'
    }

    @classmethod
    def exec(cls, f, n):
        if isinstance(n, Rule):
            x, x0 = n
            terms = 1
        elif isinstance(n, iterables):
            if len(n) != 3:
                raise FunctionException('Series::specx')
            x, x0, terms = n
            terms += 1
        else:
            raise FunctionException('Series::specx')
        try:
            if isinstance(f, iterables):
                return List(*(s.series(a, x, x0, terms) for a in f))
            return s.series(f, x, x0, terms)

        except s.PoleError or NotImplementedError as e:
            raise FunctionException('Series::sympyx', e.args[0].strip())
