import operator
from functools import reduce
import references as r
import sympy as s
from sympy.printing.pretty.stringpict import stringPict, prettyForm, xsym
from itertools import permutations, combinations
from collections.abc import Sized
from lists import List, Rule

iterables = (s.Tuple, List, Sized, s.Matrix)


class FunctionException(Exception):
    pass


# TODO: add warnings
def toList(m):
    temp_list = List()
    for row in range(m.rows):
        temp_list.append(List(m.row(row)))
    return temp_list


def thread(func, *args, **kwargs):
    """
    Internal threading function
    keyword args are not threaded
    """
    length = None

    # check list lengths
    for arg in args:
        if isinstance(arg, iterables):
            if length is not None:
                if length != len(arg):
                    raise FunctionException("Cannot Thread over Lists of unequal length.")
            else:
                length = len(iterables)

    # return called function if no lists present
    if length is None:
        return func(*args, **kwargs)

    chained = list(args)

    for i in range(len(chained)):
        if not isinstance(chained[i], iterables):
            chained[i] = (chained[i],) * length

    return List(func(*z) for z in zip(*chained))

    # if isinstance(x, iterables):
    #     temp_list = List()
    #     for item in x:
    #         temp_list.append(thread(func, item))
    #     return temp_list
    # return func(x)


def threaded(name, func):
    def fun(x):
        return thread(func, x)
    return type(name, (s.Function,), {'eval': fun})


def boolean(x):
    if isinstance(x, s.Symbol):
        if x.name == "True":
            return True
        if x.name == "False":
            return False
        return x
    return False


def assumptions(x):
    a = True
    for assumption in x:
        a = a & assumption
    return a


def in_options(arg, ops):
    if not isinstance(arg.lhs, s.Symbol) or arg.lhs.name not in ops:
        raise FunctionException(f"Unexpected option {arg.lhs}")
    return True


def options(args, ops: dict, defaults=None):
    ret = {}
    for arg in args:
        if not isinstance(arg.lhs, s.Symbol) or arg.lhs.name not in ops:
            raise FunctionException(f"Unexpected option {arg.lhs}")
        if str(arg.rhs) in ("True", "False"):
            arg.rhs = boolean(arg.rhs)
        ret[ops[arg.lhs.name]] = arg.rhs
    if defaults is not None:
        for default in defaults:
            if default not in ret:
                ret[default] = defaults[default]
    return ret


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


class ArcTan(s.Function):
    @classmethod
    def eval(cls, y, x=None):
        if x is None:
            return thread(s.atan, y)
        return thread(s.atan2, x, y)


class Exp(s.Function):
    """
    Exp [z]
     Gives the exponential of z.
    """

    @classmethod
    def eval(cls, z):
        return thread(Power, s.E, z)


class Log(s.Function):
    """
    Log [z]
     Gives the natural logarithm of z (logarithm to base e).

    Log [b, z]
     Gives the logarithm to base b.
    """

    @classmethod
    def eval(cls, x, b=None):
        if b is not None:
            return thread(s.log, b, x)
        return thread(s.log, x)


class Log2(s.Function):
    """
    Log2 [z]
     Gives the base-2 logarithm of x.
    """

    @classmethod
    def eval(cls, x):
        return thread(s.log, x, 2)


class Log10(s.Function):
    """
    Log10 [z]
     Gives the base-10 logarithm of x.
    """

    @classmethod
    def eval(cls, x):
        return thread(s.log, x, 10)


class Round(s.Function):
    """
    Round [x]
     Gives the integer closest to x.

    Round [x,a]
     Rounds to the nearest multiple of a.
    """

    @classmethod
    def eval(cls, x, a=None):
        if x.is_number:
            if a is None:
                return round(x)
            return a * round(x / a)
        if isinstance(x, iterables):
            return thread(Round, x)


class Floor(s.Function):
    """
    Floor [x]
     Gives the greatest integer less than or equal to x.

    Floor [x,a]
     Gives the greatest multiple of a less than or equal to x.

    Uses sympy.floor().
    """

    @classmethod
    def eval(cls, x, a=None):
        if a is None:
            return thread(s.floor, x)
        return thread(lambda y: a * s.floor(y / a), x)


class Ceiling(s.Function):
    """
    Ceiling [x]
     Gives the smallest integer greater than or equal to x.

    Ceiling [x, a]
     Gives the smallest multiple of a greater than or equal to x.

    Uses sympy.ceiling().
    """

    @classmethod
    def eval(cls, x, a=None):
        if a is None:
            return thread(s.ceiling, x)
        return thread(lambda y: a * s.ceiling(y / a), x)


def Min(*x):
    """
    Min [x1, {x2, x3}, x4, …]
     Gives the smallest x.
    """
    temp_list = List()
    for i in x:
        if isinstance(i, iterables):
            temp_list.append(Min(*i))
        else:
            temp_list.append(i)
    return s.Min(*temp_list)


def Max(*x):
    """
    Max [x1, {x2, x3}, x4, …]
     Gives the largest x.
    """
    temp_list = List()
    for i in x:
        if isinstance(i, iterables):
            temp_list.append(Max(*i))
        else:
            temp_list.append(i)
    return s.Max(*temp_list)


class Total(s.Function):
    """
    Total [list]
     Gives the Total Sum of elements in list.
    """

    @classmethod
    def eval(cls, _list):
        if isinstance(_list, iterables):
            return sum(_list)


class Mean(s.Function):
    """
    Mean [list]
        Gives the statistical mean of elements in list.
    """

    @classmethod
    def eval(cls, _list):
        if isinstance(_list, iterables):
            return Total(_list) / len(_list)


class Accumulate(s.Function):
    @classmethod
    def eval(cls, _list):
        temp_list = list(_list)
        if isinstance(_list, iterables):
            for i in range(1, len(_list)):
                temp_list[i] += temp_list[i - 1]
            return List(temp_list)


class Clip(s.Function):
    @classmethod
    def eval(cls, x, limits=(-1, 1)):
        if x.is_number:
            return s.Max(s.Min(x, limits[1]), limits[0])


class Quotient(s.Function):
    @classmethod
    def eval(cls, m, n):
        if m.is_number and n.is_number:
            return m // n

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real


class Rescale(s.Function):
    # TODO: clean
    @classmethod
    def eval(cls, x, x_range=None, y_range=None):
        if x_range is None and isinstance(x, iterables):
            x = list(x)
            _min = Min(x)
            _max = Max(x)
            for i in range(len(x)):
                x[i] = cls.eval(x[i], List([_min, _max]))
            return List(x)
        if isinstance(x_range, iterables) and len(x_range) == 2:
            if y_range is None or (isinstance(y_range, iterables) and len(y_range) == 2):
                if y_range is None:
                    y_range = (0, 1)
                return ((x - x_range[0]) * y_range[1] + (x_range[1] - x) * y_range[0]) / (x_range[1] - x_range[0])


class In(s.Function):
    """
    In [n]
     Gives the raw input given in the nth line.
    """
    @staticmethod
    def _in(n):
        if n is None:
            return r.refs.get_in()
        if n.is_Integer and 0 < n < r.refs.Line:
            return r.refs.get_in(n)

    @classmethod
    def eval(cls, n=None):
        return thread(cls._in, n)


class Out(s.Function):
    """
    %n
    Out [n]
     Gives the output of the nth line.

    %
        Gives the last result generated.

    %%
        Gives the result before last. %%…% (k times) gives the k^(th) previous result.
    """

    @staticmethod
    def out(n):
        out = None
        if n is None:
            out = r.refs.get_out()
        if isinstance(n, (s.Number, int, float)) and 0 < n < r.refs.Line:
            out = r.refs.get_out(n)
        if isinstance(out, s.Expr):  # TODO: Replace with Subs func.
            out = out.subs(vars(r.refs.Symbols))
        return out

    @classmethod
    def eval(cls, n=None):
        return thread(cls.out, n)


class Dot(s.Function):
    @classmethod
    def eval(cls, m, n):
        if not isinstance(m, iterables) and isinstance(n, iterables):
            return None
        m = s.Matrix(m)
        n = s.Matrix(n)

        if m.shape[1] == n.shape[1] == 1:
            return m.dot(n)
        return m * n

    def _pretty(self, printer=None):
        # TODO: redo pretty formatting
        def dot(p, *others):
            """stolen from stringpict.py"""
            if len(others) == 0:
                return p
            if p.binding > prettyForm.MUL:
                arg = stringPict(*p.parens())
            result = [p]
            for arg in others:
                result.append('.')
                if arg.binding > prettyForm.MUL:
                    arg = stringPict(*arg.parens())
                result.append(arg)
            len_res = len(result)
            for i in range(len_res):
                if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                    result.pop(i)
                    result.pop(i)
                    result.insert(i, '-')
            if result[0][0] == '-':
                bin = prettyForm.NEG
                if result[0] == '-':
                    right = result[1]
                    if right.picture[right.baseline][0] == '-':
                        result[0] = '- '
            else:
                bin = prettyForm.MUL
            return prettyForm(binding=bin, *stringPict.next(*result))

        return dot(*(printer._print(i) for i in self.args))

    def _sympystr(self, printer=None):
        return ''.join(str(i) + '.' for i in (printer.doprint(i) for i in self.args))[:-1]


class Det(s.Function):
    """
    Det [m]
     Gives the Determinant of Square Matrix m.
    """

    @classmethod
    def eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                if m.is_square:
                    return m.det()
            except ValueError:
                # TODO: Warning
                return None


class Inverse(s.Function):
    """
    Inverse [m]
     Gives the Inverse of Square Matrix m.
    """

    @classmethod
    def eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                return toList(m.inv())
            except ValueError:
                # TODO: Warning
                return None


class Transpose(s.Function):
    """
    Transpose [m]
     Gives the Transpose of Matrix m.

    Equivalent to sympy.Matrix.transpose().
    """

    @classmethod
    def eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                return toList(m.transpose())
            except ValueError:
                return None


class Re(s.Function):
    """
    Re [x]
     Gives the Real part of x.

    Equivalent to sympy.re().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.re, x)


class Im(s.Function):
    """
    Im [x]
     Gives the Imaginary part of x.

    Equivalent to sympy.im().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.im, x)


class ReIm(s.Function):
    """
    ReIm [x]
     Gives the list {Re[x], Im[x]} of x.
    """

    @classmethod
    def eval(cls, x):
        return thread(lambda b: List.create(Re(b), Im(b)), x)


class Plus(s.Function):
    @classmethod
    def eval(cls, *args):
        return reduce(operator.add, args)


class Times(s.Function):
    @classmethod
    def eval(cls, *args):
        return reduce(operator.mul, args)


class Power(s.Function):
    @classmethod
    def eval(cls, *args):
        return reduce(operator.pow, args)


class PowerMod(s.Function):
    @classmethod
    def eval(cls, a, b, m):
        return pow(a, b, m)


class Subtract(s.Function):
    @classmethod
    def eval(cls, x, y):
        return x - y


class Divide(s.Function):
    @classmethod
    def eval(cls, x, y):
        return x / y


class Abs(s.Function):
    """
    Abs [x]
     Gives the absolute value of x.

    Equivalent to sympy.Abs().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.Abs, x)


class Arg(s.Function):
    """
    Arg [x]
     Gives the argument of the complex number x.

    Equivalent to sympy.arg().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.arg, x)


class AbsArg(s.Function):
    """
    AbsArg [z]
     Gives the list {Abs[z],Arg[z]} of the number z.
    """

    @classmethod
    def eval(cls, x):
        return thread(lambda y: List.create(s.Abs(y), s.arg(y)), x)


class Factorial(s.Function):
    """
    Factorial [x]
     Gives the Factorial of x.

    Equivalent to sympy.factorial().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.factorial, x)


class Conjugate(s.Function):
    """
    Conjugate [x]
     Gives the complex conjugate of complex number x.

    Equivalent to sympy.conjugate().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.conjugate, x)


class ConjugateTranspose(s.Function):
    """
    ConjugateTranspose [m]
     Gives the conjugate transpose of m.

    Equivalent to Conjugate[Transpose[m]].
    """

    @classmethod
    def eval(cls, x):
        if isinstance(x, iterables):
            return Transpose(Conjugate(x))


class ComplexExpand(s.Function):
    """
    ComplexExpand[expr]
     Expands expr assuming that all variables are real.

    ComplexExpand [expr, {x1, x2, …}]
     Expands expr assuming that variables matching any of the x are complex.

    """

    @classmethod
    def eval(cls, x, complexes=()):
        def exp(expr):
            return s.refine(s.expand_complex(expr),
                            assumptions(s.Q.real(a) for a in expr.atoms(s.Symbol) if a not in complexes))

        if not isinstance(complexes, iterables):
            complexes = (complexes,)
        return thread(exp, x)


class LogisticSigmoid(s.Function):  # why is this here?
    """
    LogisticSigmoid [z]
     Gives the logistic sigmoid function.
    """

    @classmethod
    def eval(cls, z):
        return thread(lambda x: 1 / (1 + s.exp(-x)), z)


class Unitize(s.Function):
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
    def eval(cls, x):
        return thread(cls._unitize, x)


class Ramp(s.Function):
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
    def eval(cls, x):
        return thread(cls._ramp, x)


class Cross(s.Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], iterables) and len(args[0]) == 2:
                return List((args[0][1] * -1, args[0][0]))
        elif len(args) == 2:
            if isinstance(args[0], iterables) and isinstance(args[1], iterables):
                if len(args[0]) == len(args[1]) == 3:
                    return List(s.Matrix(args[0]).cross(s.Matrix(args[1])))

    def _pretty(self, printer=None):
        def dot(p, *others):
            """stolen from stringpict.py."""
            if len(others) == 0:
                return printer._print_Function(self, func_name='Cross')
            if p.binding > prettyForm.MUL:
                arg = stringPict(*p.parens())
            result = [p]
            for arg in others:
                result.append('×')
                if arg.binding > prettyForm.MUL:
                    arg = stringPict(*arg.parens())
                result.append(arg)
            len_res = len(result)
            for i in range(len_res):
                if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                    result.pop(i)
                    result.pop(i)
                    result.insert(i, '-')
            if result[0][0] == '-':
                bin = prettyForm.NEG
                if result[0] == '-':
                    right = result[1]
                    if right.picture[right.baseline][0] == '-':
                        result[0] = '- '
            else:
                bin = prettyForm.MUL
            return prettyForm(binding=bin, *stringPict.next(*result))

        return dot(*(printer._print(i) for i in self.args))

    def _sympystr(self, printer=None):
        return 'Cross['.join(str(i) + ', ' for i in (printer.doprint(i) for i in self.args))[:-2] + ']'


class Sign(s.Function):
    """
    Sign [x]
     Gives -1, 0, or 1 depending on whether x is negative, zero, or positive.

    For nonzero complex numbers z, Sign[z] is defined as z/Abs[z].
    """

    @staticmethod
    def _sign(n):
        if n.is_real:
            return s.sign(n)
        if n.is_complex:
            return n / Abs(n)

    @classmethod
    def eval(cls, x):
        return thread(cls._sign, x)


class Sqrt(s.Function):
    """
    Sqrt [Expr]
     Gives the Square Root of Expr.

    Equivalent to sympy.sqrt().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.sqrt, x)


# class StieltjesGamma(s.Function):
#     @classmethod
#     def eval(cls, x):
#         return thread(s.stieltjes, )


class Surd(s.Function):
    """
    Surd [x, n]
     Gives the real-valued nth root of x.

    Equivalent to sympy.real_root().
    """

    @classmethod
    def eval(cls, x, n):
        return thread(s.real_root, x, n)


class QuotientRemainder(s.Function):
    """
    QuotientRemainder [m, n]
     Gives a list of the quotient and remainder from division of m by n.
    """
    @staticmethod
    def _qr(m, n):
        return List.create(m // n, m % n)

    @classmethod
    def eval(cls, m, n):
        thread(cls._qr, m, n)


class GCD(s.Function):
    """
    GCD [x1, x2, x3, …]
     Gives the GCD of x1, x2, x3, …

    Works with Numeric and Symbolic expressions.

    Equivalent to sympy.gcd()
    """

    @classmethod
    def eval(cls, *n):
        if len(n) == 1:
            return n
        gcd = n[0]
        for number in n[1:]:
            gcd = s.gcd(gcd, number)
        return gcd


class LCM(s.Function):
    """
    LCM [x1, x2, x3, …]
     Gives the LCM of x1, x2, x3, …

    Works with Numeric and Symbolic expressions.

    Equivalent to sympy.lcm()
    """

    @classmethod
    def eval(cls, *n):
        if len(n) == 1:
            return n
        lcm = n[0]
        for number in n[1:]:
            lcm = s.lcm(lcm, number)
        return lcm


class PrimeQ(s.Function):
    """
    PrimeQ [x]
     Returns True if x is Prime.

    Equivalent to sympy.isprime().
    """

    @classmethod
    def eval(cls, n):
        return thread(s.isprime, n)


class CompositeQ(s.Function):
    """
    CompositeQ [x]
     Returns True if x is Composite.
    """

    @staticmethod
    def _comp(x):
        if x.is_number:
            if x.is_composite:
                return True
            return False

    @classmethod
    def eval(cls, n):
        return thread(cls._comp, n)


class Equal(s.Function):
    """
    Equal [x1, x2, x3, …]
     Gives a condition x1 == x2 == x3 == …
    """

    @classmethod
    def eval(cls, *args):
        if len(args) == 1:
            return None
        if len(args) == 2:
            return s.Eq(args[0], args[1])
        # TODO: better multiple equality
        return s.And(*[s.Eq(args[x], args[x + 1]) for x in range(len(args) - 1)])


class Set(s.Function):
    """
    Set [x, n]
    x = n
        Sets a symbol x to have the value n.
    """

    @classmethod
    def eval(cls, x, n):
        # TODO: Function Assignment
        for ref in [r.refs.Constants, r.refs.Functions, r.refs.Protected]:
            if str(x) in ref.__dict__:
                raise FunctionException(f'Symbol {x} cannot be Assigned to.')
        if isinstance(x, s.Symbol):
            if isinstance(n, s.Expr):
                if x in n.atoms():
                    return None
            r.refs.Symbols.__setattr__(x.name, n)
            return n
        if isinstance(x, iterables):
            if isinstance(x, iterables) and len(x) == len(n):
                return List(Set(a, b) for a, b in zip(x, n))


class Unset(s.Function):
    """
    Unset [x]
    x =.
        Deletes a symbol or list of symbols x, if they were previously assigned a value.
    """

    @classmethod
    def eval(cls, n):
        if isinstance(n, iterables):
            return List(Unset(x) for x in n)
        if isinstance(n, s.Symbol) and str(n) in r.refs.Symbols.__dict__:
            delattr(r.refs.Symbols, str(n))
        return None


class Rationalize(s.Function):
    """
    Rationalize [x]
     Converts an approximate number x to a nearby rational with small denominator.

    Rationalize [x, dx]
     Yields the rational number with smallest denominator that lies within dx of x.

    Uses sympy.nsimplify().
    See https://reference.wolfram.com/language/ref/Rationalize
    """

    @classmethod
    def eval(cls, x, dx=None):
        if isinstance(x, iterables):
            return List(Rationalize(n, dx) for n in x)
        if isinstance(x, (int, float, s.Number)):
            rat = s.nsimplify(x, rational=True, tolerance=dx)
            if dx or 1 / (10 ** 4 * s.denom(rat)) > s.Abs(rat - x):
                return rat
        return x


class Subs(s.Function):
    """
    Subs [Expr, Rule]
     Transforms Expression expr with the given Rule.

    Subs [Expr, {Rule1, Rule2, …}]
     Transforms Expression expr with the given Rules.
    """
    @classmethod
    def eval(cls, expr, replacements):
        if not isinstance(replacements, iterables):
            replacements = (replacements,)
        if isinstance(expr, iterables):
            return List(Subs(x, replacements) for x in expr)
        if isinstance(expr, s.Expr):
            expr = expr.subs(replacements)
            replacement_dict = {str(k): str(v) for k, v in replacements}
            for func in expr.atoms(s.Function):
                # TODO: sympy function replacement
                if str(func.func) in replacement_dict:
                    expr = expr.replace(func, Functions.call(replacement_dict[str(func.func)], *func.args))
            return expr


class Factor(s.Function):
    """
    Factor [Expr, Modulus -> mod, Extension -> ext, GaussianIntegers -> bool}]
     Factors an Expression.

    Equivalent to sympy.factor().
    """

    @classmethod
    def eval(cls, expr, *args):
        kws = options(args, {"Modulus": "modulus",
                             "Extension": "extension",
                             "GaussianIntegers": "gaussian"})
        return thread(s.factor, expr, **kws)


class Expand(s.Function):
    """
    Expand [Expr, Modulus -> mod]
     Expands an Expression.

    Equivalent to sympy.expand().
    """

    @classmethod
    def eval(cls, expr, *ops):
        kws = options(ops, {"Modulus": "modulus", "Trig": "trig"}, {"trig": False})
        return thread(s.expand, expr, **kws)


class TrigExpand(s.Function):
    """
    TrigExpand [Expr]
     Expands only Trigonometric Functions.

    Equivalent to sympy.expand_trig().
    """

    @classmethod
    def eval(cls, expr):
        return thread(s.expand_trig, expr)


class nPr(s.Function):
    """
    nPr [n, r]
     Gives number of possibilities for choosing an ordered set of r objects from n objects.
    """

    @staticmethod
    def _npr(x, q):
        return Factorial(x) / Factorial(x - q)

    @classmethod
    def eval(cls, n, m):
        return thread(cls._npr, n, m)


class nCr(s.Function):
    """
    nCr [n, r]
     Gives The number of different, unordered combinations of r objects from n objects.
    """

    @staticmethod
    def _ncr(x, q):
        return Factorial(x) / (Factorial(x - q) * Factorial(q))

    @classmethod
    def eval(cls, n, m):
        return thread(cls._ncr, n, m)


class N(s.Function):
    """
    N [expr]
     Gives the numerical value of expr.

    N [expr, n]
     Attempts to give a result with n-digit precision.
    """

    @classmethod
    def eval(cls, n, *args):
        return thread(lambda x: s.N(x, *args), n)


class D(s.Function):
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

    @classmethod
    def eval(cls, f, *args):
        def threaded_diff(x, *d):
            if isinstance(x, iterables):
                return List(threaded_diff(element, *d) for element in x)
            return s.diff(x, *d)

        if not args:
            return s.diff(f)

        for arg in args:
            if isinstance(arg, iterables):
                if len(arg) == 1 and isinstance(arg[0], iterables):
                    return List(threaded_diff(f, element) for element in arg[0])
                if len(arg) == 2:
                    if isinstance(arg[0], iterables):
                        f = List(threaded_diff(f, (element, arg[1])) for element in arg[0])
                    else:
                        f = threaded_diff(f, (arg[0], arg[1]))
                else:
                    # TODO: warning
                    return None
            else:
                f = threaded_diff(f, arg)
        return f


class Integrate(s.Function):
    # TODO: Doc
    @classmethod
    def eval(cls, f, *args):
        def threaded_int(x, *i):
            if isinstance(x, iterables):
                return List(threaded_int(element, *i) for element in x)
            return s.integrate(x, *i)

        if not args:
            return s.integrate(f)

        return threaded_int(f, *args)


class DiracDelta(s.Function):
    """
    DiracDelta [x]
     represents the Dirac delta function δ(x).

    DiracDelta [x1, x2, …]
     Represents the multidimensional Dirac delta function δ(x1, x2, …).

    Uses sympy.DiracDelta().
    """

    @classmethod
    def eval(cls, *args):
        return Times(*thread(s.DiracDelta, args))


class HeavisideTheta(s.Function):
    """
    HeavisideTheta [x]
     Represents the Heaviside theta function θ(x), equal to 0 for x < 0 and 1 for x > 0.

    Equivalent to sympy.Heaviside().
    """

    @classmethod
    def eval(cls, x):
        return thread(s.Heaviside, x)


class And(s.Function):
    @classmethod
    def eval(cls, *args):
        # TODO: Proper And
        return s.And(*args)


class Solve(s.Function):
    """
    Solve [expr, vars]
     Attempts to solve the system expr of equations or inequalities for the variables vars.

    Uses sympy.solve().
    """
    @classmethod
    def eval(cls, expr, v, dom=None):
        # TODO: fix (?)
        if dom is None:
            dom = s.Complexes
        if isinstance(expr, s.And):
            expr = expr.args
        # if not isinstance(expr, iterables + (s.core.relational._Inequality,)):
        #     ret = s.solveset(expr, v, dom)
        # else:
        ret = s.solve(expr, v, dict=True)
        return ret


class Simplify(s.Function):
    """
    Simplify [expr]
     Attempts to simplify the expression expr.

    Equivalent to sympy.simplify().
    """
    @classmethod
    def eval(cls, expr, assum=None):
        if assum is not None:
            raise NotImplementedError("Assumptions not implemented.")
            # if isinstance(assum, iterables):
            #     for i in range(len(assum)):
            #         if isinstance(assum[i], s.core.relational.Relational):
            #             assum[i] = s.Q.is_true(assum[i])
            #
            #     assum = assumptions(assum)
            #     expr = thread(lambda x: s.refine(x, assum), expr)
        return thread(s.simplify, expr)


class IntegerPart(s.Function):
    @staticmethod
    def integer_part(x):
        def non_complex_integer_part(n):
            if s.ask(s.Q.nonnegative(n)):
                return s.floor(n)
            return s.ceiling(n)

        if hasattr(x, "is_number") and x.is_number:
            return non_complex_integer_part(s.re(x)) + non_complex_integer_part(s.im(x)) * s.I

    @classmethod
    def eval(cls, x):
        return thread(cls.integer_part, x)


class FractionalPart(s.Function):
    @staticmethod
    def frac_part(x):
        if x.is_infinite:
            if x.is_complex or s.sign(x) > 0:
                return s.Interval(0, 1, True, True)
            return s.Interval(-1, 0, True, True)
        if hasattr(x, "is_number") and x.is_number:
            return x - IntegerPart.integer_part(x)

    @classmethod
    def eval(cls, x):
        return thread(cls.frac_part, x)


class Limit(s.Function):
    @staticmethod
    def lim(expr, lim, d='+-'):
        try:
            return s.limit(expr, lim[0], lim[1], d)
        except ValueError as e:
            if e.args[0].startswith("The limit does not exist"):
                return s.nan

    @classmethod
    def eval(cls, expr, lim, *args):
        kws = options(args, {'Direction': 'd'}, {'d': '+-'})
        d = kws['d']
        if str(d) in ("Reals", "TwoSided"):
            d = '+-'
        elif str(d) in ("FromAbove", "Right") or kws['d'] == -1:
            d = '+'
        elif str(d) in ("FromBelow", "Left") or kws['d'] == 1:
            d = '-'
        if d not in ('+', '-', '+-'):
            raise FunctionException("Invalid Limit Direction")
        return thread(lambda x: Limit.lim(x, lim, d), expr)


class Sum(s.Function):
    @classmethod
    def limits(cls, a, b):
        if isinstance(a, s.Symbol) or (isinstance(b, s.Symbol) and a.is_integer):
            return a, b
        return a, b - s.Mod(b - a, 1)

    @classmethod
    def process_skip(cls, s_, i):
        return s_.subs(i[0], i[0] * i[3]),  (i[0], *cls.limits(i[1] / i[3], i[2] / i[3]))

    @classmethod
    def process(cls, s_, i):
        if isinstance(i, iterables):
            if not isinstance(i[0], s.Symbol):
                raise FunctionException("Invalid Limits.")
            if len(i) == 2:
                return s_, (i[0], *cls.limits(s.S.One, i[1]))
            if len(i) == 3:
                return s_, (i[0], *cls.limits(i[1], i[2]))
            if len(i) == 4:
                return cls.process_skip(s_, i)
        raise FunctionException("Invalid Limits.")

    @classmethod
    def eval(cls, f, i, *xi):
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
    def eval(cls, f, i, *xi):
        i = (i,) + xi
        product = f
        for limit in reversed(i):
            if isinstance(limit, iterables):
                product = s.product(*cls.process(product, limit))
            else:
                raise NotImplementedError  # be nicer
        return product


class Zeta(s.Function):
    """
    Zeta [s]
     Gives the Riemann zeta function ζ(s).

    Zeta [s, a]
     Gives the Generalized (Hurwitz) Riemann zeta function ζ(s, a).

    Equivalent to sympy.zeta().
    """
    @classmethod
    def eval(cls, n, a=1):
        return thread(s.zeta, n, a)


class Range(s.Function):
    """
    Range [i]
     Generates the list {1, 2, …, i}.

    Range [a, b]
     Generates the list {a, …, b}.

    Range[a, b, di]
     Uses step di.
    """
    @staticmethod
    def single_range(i, n, di):
        ret = List()
        if n is None:
            n = i
            i = 1
        try:
            while (n - i) / di >= 0:
                ret.append(i)
                i += di
        except TypeError as e:
            if e.args[0].startswith('cannot determine truth value'):
                raise FunctionException('Invalid/Unsupported Range bounds.')
        return ret

    @classmethod
    def eval(cls, i, n=None, di=1):
        return thread(cls.single_range, i, n, di)


class Permutations(s.Function):
    """
    Permutations [list]
     Generates a list of all possible permutations of the elements in list.

    Permutations [list, n]
     Gives all permutations containing at most n elements.

    Permutations [list, {n}]
     Gives all permutations containing exactly n elements.

    Uses itertools.permutations().
    """
    @classmethod
    def eval(cls, li, n=None):
        if n is not None:
            if isinstance(n, iterables):
                n = Range(*n)
            else:
                if not s.Number(n).is_integer:
                    raise FunctionException("n should be an integer.")
                n = List(range(int(n) + 1))
        if isinstance(n, iterables):
            # TODO: manually remove duplicates
            ret = List()
            for i in n:
                ret = List.concat(ret, List(List(x) for x in set(permutations(li, int(i)))))
            return ret
        return List(List(x) for x in set(permutations(li, n)))


class Part(s.Function):
    @staticmethod
    def get_part(expr, n):
        if n.is_integer:
            try:
                if n > 0:
                    return expr[n - 1]
                return expr[n]
            except IndexError:
                raise FunctionException(f'Part {n} of {expr} does not exist.')
        raise FunctionException(f'{n} is not a valid Part specification.')

    @classmethod
    def eval(cls, expr, *args):
        part = None
        if not args:
            return expr
        if hasattr(expr, '__getitem__'):
            part = expr
        elif hasattr(expr, 'args'):
            part = expr.args
        if len(args) == 1:
            if args[0] == s.S.Zero:
                return s.Symbol(type(expr).__name__)
            if not part:
                raise FunctionException(f'{expr} does not have Part {args[0]}')
            return thread(cls.get_part, part, args[0])


class Table(s.Function):
    """
    Table [expr, n]
     Generates a list of n copies of expr.

    Table [expr, {i, imax}]
     Generates a list of the values of expr when i runs from 1 to imax.

    Table [expr, {i, imin, imax}]
     Starts with i = imin.

    Table [expr, {i, imin, imax, di}]
     Uses steps di.

    Table [expr, {i, {i1, i2, …}}]
     Uses the successive values i1, i2, ….
    
    Table [expr, {i, imin, imax}, {j, jmin, jmax}, …]
     Gives a nested list. The list associated with i is outermost.
    """
    @staticmethod
    def _table(expr, repl, args):
        li = List()
        for arg in args:
            li.append(Subs(expr, Rule(repl, arg)))
        return li

    @staticmethod
    def _range_parse(expr, arg):
        if arg.is_number:
            return List((expr,) * arg)
        if len(arg) == 2 and isinstance(arg[1], iterables):
            args = arg[1]
        elif len(arg) >= 2:
            args = Range(*arg[1:])
        else:
            raise FunctionException('Invalid Bounds.')  # TODO: Warning
        if not isinstance(arg[0], s.Symbol):
            raise FunctionException(f'Cannot use {arg[0]} as an Iterator.')
        return Table._table(expr, arg[0], args)

    @classmethod
    def eval(cls, expr, *args):
        if not args:
            return expr

        if len(args) == 1:
            return cls._range_parse(expr, args[0])

        li = List()
        for expr_, specs in zip(
            cls._range_parse(expr, args[0]),
            cls._range_parse(args[1:], args[0])
        ):
            li.append(Table(expr_, *specs))

        return li


class Subdivide(s.Function):
    """
    Subdivide [n]
     Generates the list {0, 1/n, 2/n, …, 1}.

    Subdivide [xmax, n]
     Generates the list of values obtained by subdividing the interval 0 to xmax into n equal parts.

    Subdivide [xmin, xmax, n]
     Generates the list of values from subdividing the interval xmin to xmax.
    """
    @classmethod
    def eval(cls, one, two=None, three=None):
        if three is None:
            if two is None:
                div = one
                x_min = 0
                x_max = 1
            else:
                x_min = 0
                x_max = one
                div = two
        else:
            x_min = one
            x_max = two
            div = three

        if div != int(div):
            raise FunctionException("Number of Subdivisions should be an Integer.")

        div = s.Number(int(div))

        step = (x_max - x_min) / div
        li = List((x_min,))

        for _ in range(int(div)):
            li.append(li[-1] + step)
        return li


class Subsets(s.Function):
    """
    Subsets [list]
     Gives a list of all possible subsets of list. (Power Set)
    
    Subsets [list, n]
     Gives all subsets containing at most n elements.

    Subsets [list, {n}]
     Gives all subsets containing exactly n elements.
    """
    @classmethod
    def eval(cls, li, n_spec=None):
        subsets = List()

        if n_spec is None:
            n_spec = range(len(li) + 1)
        elif n_spec.is_number:
            if not n_spec.is_integer:
                raise FunctionException(f'{n_spec} is not an integer.')
            n_spec = range(int(n_spec) + 1)
        else:
            n_spec = Range(*n_spec)

        for spec in n_spec:
            subsets.append(*(List(x) for x in combinations(li, spec)))

        return subsets


class FromPolarCoordinates(s.Function):
    """
    FromPolarCoordinates[{r, θ}]
     Gives the {x, y} Cartesian coordinates corresponding to the polar coordinates {r, θ}.

    FromPolarCoordinates[{r, θ1, …, θn - 2, ϕ}]
     Gives the coordinates corresponding to the hyperspherical coordinates {r, θ1, …, θn - 2, ϕ}
    """
    @classmethod
    def eval(cls, list_):
        # TODO: Thread
        length = len(list_)
        if length == 1:
            raise FunctionException('Polar Coordinates can only be defined for dimesions of 2 and greater.')
        ret = List(list_[:1] * length)
        for pos, angle in enumerate(list_[1:]):
            ret[pos] *= s.cos(angle)
            for x in range(pos + 1, length):
                ret[x] *= s.sin(angle)
        return ret


class ToPolarCoordinates(s.Function):
    """
    ToPolarCoordinates[{x, y}]
     Gives the {r, θ} polar coordinates corresponding to the Cartesian coordinates {x, y}.

    ToPolarCoordinates[{x1, x2, …, xn}]
     Gives the hyperspherical coordinates corresponding to the Cartesian coordinates {x1, x2, …, xn}.
    """
    @classmethod
    def eval(cls, list_):
        # TODO: Thread
        list_ = List(list_)
        length = len(list_)
        if length == 1:
            raise FunctionException('Polar Coordinates can only be defined for dimesions of 2 and greater.')
        ret = List.create(Sqrt(Total(list_ ** 2)))
        for x in range(length - 2):
            ret.append(s.acos(list_[x] / Sqrt(Total(list_[x:] ** 2))))
        ret.append(ArcTan(list_[-2], list_[-1]))
        return ret


class Functions:
    # TODO: Move functions into class (not doing that/finding a better solution was naive)

    # TODO: Subs List replacement

    # TODO: Part, Span
    # TODO: Semicolon
    # TODO: Logical Or

    # TODO: Polar Complex Number Representation
    # TODO: Series
    # TODO: DSolve
    # TODO: Series
    # TODO: Random Functions

    # TODO: Simple List Functions
    # TODO: Nothing (Lists)

    # TODO: Matrix Representation
    # TODO: Matrix Row Operations
    # TODO: Remaining Matrix Operations

    # TODO: Warnings

    # TODO: Function Assisgnment
    # TODO: Subs Function Replacement

    # TODO: Latex Printer
    # TODO: References Storage

    # Low Priority

    # TODO: Map, Apply
    # TODO: Pretty Printer Fixes for Dot, Cross
    # TODO: Arithmetic Functions: Ratios, Differences
    # TODO: Booleans, Conditions, Boole
    # TODO: Cross of > 3 dimensional vectors
    # TODO: Implement Fully: Total, Clip, Quotient, Mod, Factor

    # for now, until I find something better
    Functions = {k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, s.Function)}

    @classmethod
    def call(cls, f, *a):
        if f in r.refs.NoCache:
            s.core.cache.clear_cache()
        if f in cls.Functions:
            return cls.Functions[f](*a)
        return s.Function(f)(*a)
