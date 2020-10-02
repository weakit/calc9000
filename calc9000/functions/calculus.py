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

    Uses sympy.diff().
    """

    tags = {
        'argx': 'No variable of differentiation was specified to differentiate a multi-variate expression.',
        'spec': 'Invalid specification for D.'
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
                raise FunctionException('D::argx')

        for arg in args:
            if isinstance(arg, iterables):
                if len(arg) == 1 and isinstance(arg[0], iterables):
                    return List.create(threaded_diff(f, element) for element in arg[0])
                if len(arg) == 2:
                    if isinstance(arg[0], iterables):
                        f = List.create(threaded_diff(f, (element, arg[1])) for element in arg[0])
                    else:
                        f = threaded_diff(f, (arg[0], arg[1]))
                else:
                    raise FunctionException('D::spec')
            else:
                f = threaded_diff(f, arg)
        return f


class Integrate(NormalFunction):
    # TODO: Doc

    tags = {
        'argx': 'No variable of integration was specified to integrate a multi-variate expression.'
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
                raise FunctionException('Integrate::argx')

        return threaded_int(f, *args)


class Limit(NormalFunction):
    tags = {
        'lim': 'Invalid Limit.',
        'dir': 'Invalid Limit Direction.'
    }

    @staticmethod
    def lim(expr, lim, d='+-'):
        if not isinstance(lim, Rule):
            raise FunctionException('Limit::lim')
        try:
            return s.limit(expr, lim.lhs, lim.rhs, d)
        except ValueError as e:
            if e.args[0].startswith("The limit does not exist"):
                return s.nan

    op_spec = ({'Direction': 'd'}, {'d': '+-'})
    param_spec = (2, 2)
    rule_param = True

    @classmethod
    def exec(cls, expr, lim, d='+-'):
        if isinstance(d, String):
            if d.value in ("Reals", "TwoSided"):
                d = '+-'
            elif d.value in ("FromAbove", "Right") or d == -1:
                d = '+'
            elif d.value in ("FromBelow", "Left") or d == 1:
                d = '-'
        elif is_integer(d):
            if d == -1:
                d = '+'
            elif d == 1:
                d = '-'
        if d not in ('+', '-', '+-'):
            raise FunctionException('Limit::dir')
        return thread(lambda x: Limit.lim(x, lim, d), expr)
