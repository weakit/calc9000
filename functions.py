import operator as op
from functools import reduce
import references as r
import sympy as s
from sympy.printing.pretty.stringpict import stringPict, prettyForm, xsym
from itertools import permutations, combinations
from collections.abc import Sized
from datatypes import List, Rule

iterables = (s.Tuple, List, Sized, s.Matrix, list, tuple)


class FunctionException(Exception):
    pass


class NormalFunction(s.Function):
    """
    Ordinary Function Class
    Works for most Functions.
    """
    @staticmethod
    def _make_replacements(x: s.Basic):
        # if hasattr(x, 'subs'):
        #     return x.subs(r.refs.Symbols)
        return x

    @classmethod
    def eval(cls, *args):
        return cls.Eval(*(cls._make_replacements(x) for x in args))


class PilotFunction(s.Function):
    """
    Placeholder Function Class.
    Acts as an Unevaluated Function.
    """
    @staticmethod
    def land_in(expr):
        for x in expr.atoms(PilotFunction)[:]:
            expr = expr.subs(x, x.land())
        return expr

    def land(self):
        return Functions.call(type(self).__name__, *self.args)


# TODO: add warnings
def toList(m):
    temp_list = List()
    for row in range(m.rows):
        temp_list.append(List(*m.row(row)))
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

    return List.create(func(*z) for z in zip(*chained))

    # if isinstance(x, iterables):
    #     temp_list = List()
    #     for item in x:
    #         temp_list.append(thread(func, item))
    #     return temp_list
    # return func(x)


def threaded(name, func):
    def fun(x):
        return thread(func, x)
    return type(name, (NormalFunction,), {'eval': fun})


def boolean(x):
    if isinstance(x, s.Symbol):
        if x.name == "True":
            return True
        if x.name == "False":
            return False
        return x
    return False


def ands(x):
    a = True
    for and_ in x:
        a = a & and_
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


def is_integer(n):
    if hasattr(n, 'is_integer'):
        return bool(n.is_integer)
    if hasattr(n, 'is_Integer'):
        return bool(n.is_integer)
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return int(n) == n
    return False


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
    def Eval(cls, y, x=None):
        if x is None:
            return thread(s.atan, y)
        return thread(s.atan2, x, y)


class Exp(NormalFunction):
    """
    Exp [z]
     Gives the exponential of z.
    """

    @classmethod
    def Eval(cls, z):
        return thread(Power, s.E, z)


class Log(NormalFunction):
    """
    Log [z]
     Gives the natural logarithm of z (logarithm to base e).

    Log [b, z]
     Gives the logarithm to base b.
    """

    @classmethod
    def Eval(cls, x, b=None):
        if b is not None:
            return thread(s.log, b, x)
        return thread(s.log, x)


class Log2(NormalFunction):
    """
    Log2 [z]
     Gives the base-2 logarithm of x.
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.log, x, 2)


class Log10(NormalFunction):
    """
    Log10 [z]
     Gives the base-10 logarithm of x.
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.log, x, 10)


class Round(NormalFunction):
    """
    Round [x]
     Gives the integer closest to x.

    Round [x,a]
     Rounds to the nearest multiple of a.
    """

    @classmethod
    def Eval(cls, x, a=None):
        if x.is_number:
            if a is None:
                return round(x)
            return a * round(x / a)
        if isinstance(x, iterables):
            return thread(Round, x)


class Floor(NormalFunction):
    """
    Floor [x]
     Gives the greatest integer less than or equal to x.

    Floor [x,a]
     Gives the greatest multiple of a less than or equal to x.

    Uses sympy.floor().
    """

    @classmethod
    def Eval(cls, x, a=None):
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
    def Eval(cls, x, a=None):
        if a is None:
            return thread(s.ceiling, x)
        return thread(lambda y: a * s.ceiling(y / a), x)


class Min(NormalFunction):
    """
    Min [x1, {x2, x3}, x4, …]
     Gives the smallest x.
    """
    @classmethod
    def Eval(cls, *x):
        temp_list = List()
        for i in x:
            if isinstance(i, iterables):
                temp_list.append(Min(*i))
            else:
                temp_list.append(i)
        return s.Min(*temp_list)


class Max(NormalFunction):
    """
    Max [x1, {x2, x3}, x4, …]
     Gives the largest x.
    """
    @classmethod
    def Eval(cls, *x):
        temp_list = List()
        for i in x:
            if isinstance(i, iterables):
                temp_list.append(Max(*i))
            else:
                temp_list.append(i)
        return s.Max(*temp_list)


class Total(NormalFunction):
    """
    Total [list]
     Gives the Total Sum of elements in list.
    """

    @classmethod
    def Eval(cls, _list):
        if isinstance(_list, iterables):
            return sum(_list)


class Mean(NormalFunction):
    """
    Mean [list]
        Gives the statistical mean of elements in list.
    """

    @classmethod
    def Eval(cls, _list):
        if isinstance(_list, iterables):
            return Total(_list) / len(_list)


class Accumulate(NormalFunction):
    @classmethod
    def Eval(cls, _list):
        temp_list = list(_list)
        if isinstance(_list, iterables):
            for i in range(1, len(_list)):
                temp_list[i] += temp_list[i - 1]
            return List(*temp_list)


class Clip(NormalFunction):
    @classmethod
    def Eval(cls, x, limits=(-1, 1)):
        if x.is_number:
            return s.Max(s.Min(x, limits[1]), limits[0])


class Quotient(NormalFunction):
    @classmethod
    def Eval(cls, m, n):
        if m.is_number and n.is_number:
            return m // n

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real


class Rescale(NormalFunction):
    # TODO: clean
    @classmethod
    def Eval(cls, x, x_range=None, y_range=None):
        if x_range is None and isinstance(x, iterables):
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


class In(NormalFunction):
    """
    In [n]
     Gives the raw input given in the nth line.
    """
    @staticmethod
    def _in(n):
        if n is None:
            return r.refs.get_in()
        if is_integer(n) and 0 < n < r.refs.Line:
            return r.refs.get_in(n)

    @classmethod
    def Eval(cls, n=None):
        return thread(cls._in, n)


class Out(NormalFunction):
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
            out = out.subs(r.refs.Symbols)
        return out

    @classmethod
    def Eval(cls, n=None):
        return thread(cls.out, n)


class Dot(NormalFunction):
    @classmethod
    def Eval(cls, m, n):
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


class Det(NormalFunction):
    """
    Det [m]
     Gives the Determinant of Square Matrix m.
    """

    @classmethod
    def Eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                if m.is_square:
                    return m.det()
            except ValueError:
                # TODO: Warning
                return None


class Inverse(NormalFunction):
    """
    Inverse [m]
     Gives the Inverse of Square Matrix m.
    """

    @classmethod
    def Eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                return toList(m.inv())
            except ValueError:
                # TODO: Warning
                return None


class Transpose(NormalFunction):
    """
    Transpose [m]
     Gives the Transpose of Matrix m.

    Equivalent to sympy.Matrix.transpose().
    """

    @classmethod
    def Eval(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                return toList(m.transpose())
            except ValueError:
                return None


class Re(NormalFunction):
    """
    Re [x]
     Gives the Real part of x.

    Equivalent to sympy.re().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.re, x)


class Im(NormalFunction):
    """
    Im [x]
     Gives the Imaginary part of x.

    Equivalent to sympy.im().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.im, x)


class ReIm(NormalFunction):
    """
    ReIm [x]
     Gives the list {Re[x], Im[x]} of x.
    """

    @classmethod
    def Eval(cls, x):
        return thread(lambda b: List(Re(b), Im(b)), x)


class Plus(NormalFunction):
    @classmethod
    def Eval(cls, *args):
        return reduce(op.add, args)


class Times(NormalFunction):
    @classmethod
    def Eval(cls, *args):
        return reduce(op.mul, args)


class Power(NormalFunction):
    @classmethod
    def Eval(cls, *args):
        return reduce(op.pow, args)


class PowerMod(NormalFunction):
    @classmethod
    def Eval(cls, a, b, m):
        return pow(a, b, m)


class Subtract(NormalFunction):
    @classmethod
    def Eval(cls, x, y):
        return x - y


class Divide(NormalFunction):
    @classmethod
    def Eval(cls, x, y):
        return x / y


class Abs(NormalFunction):
    """
    Abs [x]
     Gives the absolute value of x.

    Equivalent to sympy.Abs().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.Abs, x)


class Arg(NormalFunction):
    """
    Arg [x]
     Gives the argument of the complex number x.

    Equivalent to sympy.arg().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.arg, x)


class AbsArg(NormalFunction):
    """
    AbsArg [z]
     Gives the list {Abs[z],Arg[z]} of the number z.
    """

    @classmethod
    def Eval(cls, x):
        return thread(lambda y: List(s.Abs(y), s.arg(y)), x)


class Factorial(NormalFunction):
    """
    Factorial [x]
     Gives the Factorial of x.

    Equivalent to sympy.factorial().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.factorial, x)


class Conjugate(NormalFunction):
    """
    Conjugate [x]
     Gives the complex conjugate of complex number x.

    Equivalent to sympy.conjugate().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.conjugate, x)


class ConjugateTranspose(NormalFunction):
    """
    ConjugateTranspose [m]
     Gives the conjugate transpose of m.

    Equivalent to Conjugate[Transpose[m]].
    """

    @classmethod
    def Eval(cls, x):
        if isinstance(x, iterables):
            return Transpose(Conjugate(x))


class ComplexExpand(NormalFunction):
    """
    ComplexExpand[expr]
     Expands expr assuming that all variables are real.

    ComplexExpand [expr, {x1, x2, …}]
     Expands expr assuming that variables matching any of the x are complex.

    """

    @classmethod
    def Eval(cls, x, complexes=()):
        def exp(expr):
            return s.refine(s.expand_complex(expr),
                            ands(s.Q.real(a) for a in expr.atoms(s.Symbol) if a not in complexes))

        if not isinstance(complexes, iterables):
            complexes = (complexes,)
        return thread(exp, x)


class LogisticSigmoid(NormalFunction):  # why is this here?
    """
    LogisticSigmoid [z]
     Gives the logistic sigmoid function.
    """

    @classmethod
    def Eval(cls, z):
        return thread(lambda x: 1 / (1 + s.exp(-x)), z)


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
    def Eval(cls, x):
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
    def Eval(cls, x):
        return thread(cls._ramp, x)


class Cross(NormalFunction):
    @classmethod
    def Eval(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], iterables) and len(args[0]) == 2:
                return List(args[0][1] * -1, args[0][0])
        elif len(args) == 2:
            if isinstance(args[0], iterables) and isinstance(args[1], iterables):
                if len(args[0]) == len(args[1]) == 3:
                    return List.create(s.Matrix(args[0]).cross(s.Matrix(args[1])))

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


class Sign(NormalFunction):
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
    def Eval(cls, x):
        return thread(cls._sign, x)


class Sqrt(NormalFunction):
    """
    Sqrt [Expr]
     Gives the Square Root of Expr.

    Equivalent to sympy.sqrt().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.sqrt, x)


# class StieltjesGamma(NormalFunction):
#     @classmethod
#     def Eval(cls, x):
#         return thread(s.stieltjes, )


class Surd(NormalFunction):
    """
    Surd [x, n]
     Gives the real-valued nth root of x.

    Equivalent to sympy.real_root().
    """

    @classmethod
    def Eval(cls, x, n):
        return thread(s.real_root, x, n)


class QuotientRemainder(NormalFunction):
    """
    QuotientRemainder [m, n]
     Gives a list of the quotient and remainder from division of m by n.
    """
    @staticmethod
    def _qr(m, n):
        return List(m // n, m % n)

    @classmethod
    def Eval(cls, m, n):
        thread(cls._qr, m, n)


class GCD(NormalFunction):
    """
    GCD [x1, x2, x3, …]
     Gives the GCD of x1, x2, x3, …

    Works with Numeric and Symbolic expressions.

    Equivalent to sympy.gcd()
    """

    @classmethod
    def Eval(cls, *n):
        if len(n) == 1:
            return n
        gcd = n[0]
        for number in n[1:]:
            gcd = s.gcd(gcd, number)
        return gcd


class LCM(NormalFunction):
    """
    LCM [x1, x2, x3, …]
     Gives the LCM of x1, x2, x3, …

    Works with Numeric and Symbolic expressions.

    Equivalent to sympy.lcm()
    """

    @classmethod
    def Eval(cls, *n):
        if len(n) == 1:
            return n
        lcm = n[0]
        for number in n[1:]:
            lcm = s.lcm(lcm, number)
        return lcm


class PrimeQ(NormalFunction):
    """
    PrimeQ [x]
     Returns True if x is Prime.

    Equivalent to sympy.isprime().
    """

    @classmethod
    def Eval(cls, n):
        return thread(s.isprime, n)


class CompositeQ(NormalFunction):
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
    def Eval(cls, n):
        return thread(cls._comp, n)


class Equal(NormalFunction):
    """
    Equal [x1, x2, x3, …]
     Gives a condition x1 == x2 == x3 == …
    """

    @classmethod
    def Eval(cls, *args):
        if len(args) == 1:
            return None
        if len(args) == 2:
            return s.Eq(args[0], args[1])
        # TODO: better multiple equality
        return s.And(*[s.Eq(args[x], args[x + 1]) for x in range(len(args) - 1)])


class Set(NormalFunction):
    """
    Set [x, n]
     x = n
     Sets a symbol x to have the value n.
    """

    @classmethod
    def Eval(cls, x, n):
        refs = r.refs
        for ref in [
            refs.Constants.__dict__,
            refs.BuiltIns,
            refs.Protected.__dict__
        ]:
            if str(x) in ref:
                raise FunctionException(f'Symbol {x} cannot be Assigned to.')
        if isinstance(x, s.Symbol):
            if isinstance(n, s.Expr):
                if x in n.atoms():
                    return None
            refs.Symbols[x.name] = n
            return n
        if isinstance(x, s.Function):
            pass
            # TODO: think again
            # list_ = []
            # name = type(x).__name__
            # expr = n
            # num = 1
            # for arg in x.args:
            #     if isinstance(arg, s.Symbol) and arg.name.endswith('_'):
            #         expr = Subs(expr, Rule(s.Symbol(arg.name[:-1]), s.Symbol(f'*{num}')))
            #         list_.append(s.Symbol(f'*{num}'))
            #         num += 1
            #     else:
            #         list_.append(arg)
            # if name not in refs.Functions:
            #     refs.Functions[name] = {tuple(list_): (expr, ())}
            # else:
            #     refs.Functions[name].update({tuple(list_): (expr, ())})
            # return n
        if isinstance(x, iterables):
            if isinstance(x, iterables) and len(x) == len(n):
                return List.create(Set(a, b) for a, b in zip(x, n))


def DelayedSet(f, x, n):
    # TODO: again
    refs = r.refs
    for ref in [
        refs.Constants.__dict__,
        refs.BuiltIns,
        refs.Protected.__dict__
    ]:
        if str(x) in ref:
            raise FunctionException(f'Symbol {x} cannot be Assigned to.')
    if isinstance(x, s.Symbol):
        if isinstance(n, s.Expr):
            if x in n.atoms():
                return None
        refs.Symbols[x.name] = n
        return n
    if isinstance(x, s.Function):
        list_ = []
        name = type(x).__name__
        expr = n
        num = 1
        for arg in x.args:
            if isinstance(arg, s.Symbol) and arg.name.endswith('_'):
                expr = Subs(expr, Rule(s.Symbol(arg.name[:-1]), s.Symbol(f'*{num}')))
                list_.append(s.Symbol(f'*{num}'))
                num += 1
            else:
                list_.append(arg)
        if name not in refs.Functions:
            refs.Functions[name] = {tuple(list_): (expr, f)}
        else:
            refs.Functions[name].update({tuple(list_): (expr, f)})
        return n
    if isinstance(x, iterables):
        if isinstance(x, iterables) and len(x) == len(n):
            return List.create(DelayedSet(f, a, b) for a, b in zip(x, n))


class Unset(NormalFunction):
    """
    Unset [x]
    x =.
        Deletes a symbol or list of symbols x, if they were previously assigned a value.
    """

    @classmethod
    def Eval(cls, n):
        if isinstance(n, iterables):
            return List.create(Unset(x) for x in n)
        if isinstance(n, s.Symbol) and str(n) in r.refs.Symbols:
            del r.refs.Symbols[str(n)]
        return None


class Rationalize(NormalFunction):
    """
    Rationalize [x]
     Converts an approximate number x to a nearby rational with small denominator.

    Rationalize [x, dx]
     Yields the rational number with smallest denominator that lies within dx of x.

    Uses sympy.nsimplify().
    See https://reference.wolfram.com/language/ref/Rationalize
    """

    @classmethod
    def Eval(cls, x, dx=None):
        if isinstance(x, iterables):
            return List.create(Rationalize(n, dx) for n in x)
        if isinstance(x, (int, float, s.Number)):
            rat = s.nsimplify(x, rational=True, tolerance=dx)
            if dx or 1 / (10 ** 4 * s.denom(rat)) > s.Abs(rat - x):
                return rat
        return x


class Subs(NormalFunction):
    """
    Subs [Expr, Rule]
     Transforms Expression expr with the given Rule.

    Subs [Expr, {Rule1, Rule2, …}]
     Transforms Expression expr with the given Rules.
    """
    @classmethod
    def Eval(cls, expr, replacements):
        if not isinstance(replacements, iterables):
            replacements = (replacements,)
        if isinstance(expr, iterables):
            return List.create(Subs(x, replacements) for x in expr)
        if isinstance(expr, s.Expr):
            expr = expr.subs(replacements)
            replacement_dict = {str(k): str(v) for k, v in replacements}
            for func in expr.atoms(NormalFunction):
                # TODO: sympy function replacement
                if str(func.func) in replacement_dict:
                    expr = expr.replace(func, Functions.call(replacement_dict[str(func.func)], *func.args))
            return expr


class Factor(NormalFunction):
    """
    Factor [Expr, Modulus -> mod, Extension -> ext, GaussianIntegers -> bool}]
     Factors an Expression.

    Equivalent to sympy.factor().
    """

    @classmethod
    def Eval(cls, expr, *args):
        kws = options(args, {"Modulus": "modulus",
                             "Extension": "extension",
                             "GaussianIntegers": "gaussian"})
        return thread(s.factor, expr, **kws)


class Expand(NormalFunction):
    """
    Expand [Expr, Modulus -> mod]
     Expands an Expression.

    Equivalent to sympy.expand().
    """

    @classmethod
    def Eval(cls, expr, *ops):
        kws = options(ops, {"Modulus": "modulus", "Trig": "trig"}, {"trig": False})
        return thread(s.expand, expr, **kws)


class TrigExpand(NormalFunction):
    """
    TrigExpand [Expr]
     Expands only Trigonometric Functions.

    Equivalent to sympy.expand_trig().
    """

    @classmethod
    def Eval(cls, expr):
        return thread(s.expand_trig, expr)


class nPr(NormalFunction):
    """
    nPr [n, r]
     Gives number of possibilities for choosing an ordered set of r objects from n objects.
    """

    @staticmethod
    def _npr(x, q):
        return Factorial(x) / Factorial(x - q)

    @classmethod
    def Eval(cls, n, m):
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
    def Eval(cls, n, m):
        return thread(cls._ncr, n, m)


class N(NormalFunction):
    """
    N [expr]
     Gives the numerical value of expr.

    N [expr, n]
     Attempts to give a result with n-digit precision.
    """

    @classmethod
    def Eval(cls, n, *args):
        return thread(lambda x: s.N(x, *args), n)


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

    @classmethod
    def Eval(cls, f, *args):
        def threaded_diff(x, *d):
            if isinstance(x, iterables):
                return List.create(threaded_diff(element, *d) for element in x)
            return s.diff(x, *d)

        if not args:
            return s.diff(f)

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
                    # TODO: warning
                    return None
            else:
                f = threaded_diff(f, arg)
        return f


class Integrate(NormalFunction):
    # TODO: Doc
    @classmethod
    def Eval(cls, f, *args):
        def threaded_int(x, *i):
            if isinstance(x, iterables):
                return List.create(threaded_int(element, *i) for element in x)
            return s.integrate(x, *i)

        if not args:
            return s.integrate(f)

        return threaded_int(f, *args)


class DiracDelta(NormalFunction):
    """
    DiracDelta [x]
     represents the Dirac delta function δ(x).

    DiracDelta [x1, x2, …]
     Represents the multidimensional Dirac delta function δ(x1, x2, …).

    Uses sympy.DiracDelta().
    """

    @classmethod
    def Eval(cls, *args):
        return Times(*thread(s.DiracDelta, args))


class HeavisideTheta(NormalFunction):
    """
    HeavisideTheta [x]
     Represents the Heaviside theta function θ(x), equal to 0 for x < 0 and 1 for x > 0.

    Equivalent to sympy.Heaviside().
    """

    @classmethod
    def Eval(cls, x):
        return thread(s.Heaviside, x)


class And(NormalFunction):
    @classmethod
    def Eval(cls, *args):
        # TODO: Proper And
        return s.And(*args)


class Solve(NormalFunction):
    """
    Solve [expr, vars]
     Attempts to solve the system expr of equations or inequalities for the variables vars.

    Uses sympy.solve().
    """
    @classmethod
    def Eval(cls, expr, v, dom=None):
        # TODO: fix (?)
        if dom is None:
            dom = s.Complexes
        if isinstance(expr, s.And):
            expr = expr.args
        # if not isinstance(expr, iterables + (s.core.relational._Inequality,)):
        #     ret = s.solveset(expr, v, dom)
        # else:
        # if isinstance(expr, iterables):
        #     expr = ands(expr)
        ret = s.solve(expr, v, dict=True)
        return ret


class Simplify(NormalFunction):
    """
    Simplify [expr]
     Attempts to simplify the expression expr.

    Equivalent to sympy.simplify().
    """
    @classmethod
    def Eval(cls, expr, assum=None):
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
    def Eval(cls, x):
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

    @classmethod
    def Eval(cls, x):
        return thread(cls.frac_part, x)


class Limit(NormalFunction):
    @staticmethod
    def lim(expr, lim, d='+-'):
        try:
            return s.limit(expr, lim[0], lim[1], d)
        except ValueError as e:
            if e.args[0].startswith("The limit does not exist"):
                return s.nan

    @classmethod
    def Eval(cls, expr, lim, *args):
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


class Sum(NormalFunction):
    @classmethod
    def limits(cls, a, b):
        if isinstance(a, s.Symbol) or (isinstance(b, s.Symbol) and is_integer(a)):
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
    def Eval(cls, f, i, *xi):
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
    def Eval(cls, f, i, *xi):
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
     Gives the Generalized (Hurwitz) Riemann zeta function ζ(s, a).

    Equivalent to sympy.zeta().
    """
    @classmethod
    def Eval(cls, n, a=1):
        return thread(s.zeta, n, a)


class Range(NormalFunction):
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
    def Eval(cls, i, n=None, di=1):
        return thread(cls.single_range, i, n, di)


class Permutations(NormalFunction):
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
    def Eval(cls, li, n=None):
        if n is not None:
            if isinstance(n, iterables):
                n = Range(*n)
            else:
                if not is_integer(n):
                    raise FunctionException("n should be an integer.")
                n = List.create(range(int(n) + 1))
        if isinstance(n, iterables):
            # TODO: manually remove duplicates
            ret = List()
            for i in n:
                ret = List.concat(ret, List.create(List.create(x) for x in set(permutations(li, int(i)))))
            return ret
        return List.create(List.create(x) for x in set(permutations(li, n)))


class Table(NormalFunction):
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
            return List.create((expr,) * arg)
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
    def Eval(cls, expr, *args):
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


class Subdivide(NormalFunction):
    """
    Subdivide [n]
     Generates the list {0, 1/n, 2/n, …, 1}.

    Subdivide [xmax, n]
     Generates the list of values obtained by subdividing the interval 0 to xmax into n equal parts.

    Subdivide [xmin, xmax, n]
     Generates the list of values from subdividing the interval xmin to xmax.
    """
    @classmethod
    def Eval(cls, one, two=None, three=None):
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

        if not is_integer(div):
            raise FunctionException("Number of Subdivisions should be an Integer.")

        div = s.Number(int(div))

        step = (x_max - x_min) / div
        li = List(x_min)

        for _ in range(int(div)):
            li.append(li[-1] + step)
        return li


class Subsets(NormalFunction):
    """
    Subsets [list]
     Gives a list of all possible subsets of list. (Power Set)
    
    Subsets [list, n]
     Gives all subsets containing at most n elements.

    Subsets [list, {n}]
     Gives all subsets containing exactly n elements.
    """
    @classmethod
    def Eval(cls, li, n_spec=None):
        subsets = List()

        if n_spec is None:
            n_spec = range(len(li) + 1)
        elif n_spec.is_number:
            if not is_integer(n_spec):
                raise FunctionException(f'{n_spec} is not an integer.')
            n_spec = range(int(n_spec) + 1)
        else:
            n_spec = Range(*n_spec)

        for spec in n_spec:
            subsets.append(*(List.create(x) for x in combinations(li, spec)))

        return subsets


class FromPolarCoordinates(NormalFunction):
    """
    FromPolarCoordinates[{r, θ}]
     Gives the {x, y} Cartesian coordinates corresponding to the polar coordinates {r, θ}.

    FromPolarCoordinates[{r, θ1, …, θn - 2, ϕ}]
     Gives the coordinates corresponding to the hyperspherical coordinates {r, θ1, …, θn - 2, ϕ}
    """
    @classmethod
    def Eval(cls, list_):
        # TODO: Thread
        length = len(list_)
        if length == 1:
            raise FunctionException('Polar Coordinates can only be defined for dimesions of 2 and greater.')
        ret = List.create(list_[:1] * length)
        for pos, angle in enumerate(list_[1:]):
            ret[pos] *= s.cos(angle)
            for x in range(pos + 1, length):
                ret[x] *= s.sin(angle)
        return ret


class ToPolarCoordinates(NormalFunction):
    """
    ToPolarCoordinates[{x, y}]
     Gives the {r, θ} polar coordinates corresponding to the Cartesian coordinates {x, y}.

    ToPolarCoordinates[{x1, x2, …, xn}]
     Gives the hyperspherical coordinates corresponding to the Cartesian coordinates {x1, x2, …, xn}.
    """
    @classmethod
    def Eval(cls, list_):
        # TODO: Thread
        list_ = List(*list_)
        length = len(list_)
        if length == 1:
            raise FunctionException('Polar Coordinates can only be defined for dimesions of 2 and greater.')
        ret = List(Sqrt(Total(list_ ** 2)))
        for x in range(length - 2):
            ret.append(s.acos(list_[x] / Sqrt(Total(list_[x:] ** 2))))
        ret.append(ArcTan(list_[-2], list_[-1]))
        return ret


class Together(NormalFunction):
    @classmethod
    def Eval(cls, expr):
        return thread(lambda x: s.simplify(s.together(x)), expr)


class Apart(NormalFunction):
    @classmethod
    def Eval(cls, expr, x=None):
        return thread(s.apart, expr, x)


class Part(NormalFunction):
    @staticmethod
    def get_part(expr, n):
        if not isinstance(expr, iterables):
            # TODO: Associations and other Heads
            raise NotImplementedError
        if is_integer(n):
            try:
                if n > 0:
                    return expr[int(n - 1)]
                return expr[int(n)]
            except IndexError:
                raise FunctionException(f'Part {n} of {expr} does not exist.')
        raise FunctionException(f'{n} is not a valid Part specification.')

    @classmethod
    def Eval(cls, expr, *args):
        part = None
        if not args:
            return expr
        if hasattr(expr, 'args'):
            part = expr.args
        elif hasattr(expr, '__getitem__'):
            part = expr
        arg = args[0]
        if arg == s.S.Zero:
            return s.Symbol(type(expr).__name__)
        if not part:
            raise FunctionException(f'{expr} does not have Part {arg}')
        if arg == r.refs.Constants.All:
            arg = Range(len(expr))
        if isinstance(arg, iterables):
            return List.create(Part(cls.get_part(part, x), *args[1:]) for x in arg)
        return Part(cls.get_part(part, arg), *args[1:])


class Take(NormalFunction):
    @staticmethod
    def ul(upper, lower):
        # I don't know how to do this better
        if lower > 0:
            lower -= 1
        if upper < 0:
            upper += 1
        if upper == 0:
            upper = None
        if lower == 0:
            lower = None
        return upper, lower

    @classmethod
    def Eval(cls, expr, *seqs):
        take = head = None

        if not seqs:
            return expr
        if hasattr(expr, 'args'):
            take = expr.args
            head = expr.__class__
        elif hasattr(expr, '__getitem__'):
            take = expr
            head = List

        for seq in seqs:
            if isinstance(seq, iterables):
                if len(seq) == 1:
                    return Part(take, seq)
                lower = seq[0]
                upper = seq[1]
                step = 1
                if 0 in (lower, upper, step) or not (is_integer(lower) and is_integer(upper) and is_integer(step)):
                    raise FunctionException('Invalid Bounds for Take.')
                if len(seq) == 3:
                    step = seq[2]
                if step > 0:
                    upper, lower = cls.ul(upper, lower)
                else:
                    upper -= 1
                    lower, upper = cls.ul(lower, upper)
                return head(*take[lower:upper:step])
            if is_integer(seq):
                if seq > 0:
                    return head(*take[:seq])
                return head(*take[seq:])
            else:
                raise FunctionException(f'{seq} is not a valid Take specification.')


class Functions:
    # TODO: Move functions into class (not doing that/finding a better solution was naive)

    # TODO: Subs List replacement

    # TODO: Collect
    # TODO: Span
    # TODO: Prime Notation
    # TODO: Part, Assignment
    # TODO: Semicolon
    # TODO: Logical Or

    # TODO: Polar Complex Number Representation
    # TODO: Series
    # TODO: Solve output
    # TODO: NSolve, DSolve
    # TODO: Roots (Solve)
    # TODO: Series
    # TODO: Random Functions
    # TODO: Unit Conversions

    # TODO: Simple List Functions
    # TODO: Nothing (Lists)

    # TODO: Make Matrix Functions use List
    # TODO: Matrix Representation
    # TODO: Matrix Row Operations
    # TODO: Remaining Matrix Operations

    # TODO: Warnings

    # TODO: Clear Function from References
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
    r.refs.BuiltIns.update({k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, s.Function)})

    @classmethod
    def not_normal(cls, f: str):
        if f in r.refs.BuiltIns:
            return not issubclass(r.refs.BuiltIns[f], NormalFunction)
        return False

    @classmethod
    def call(cls, f: str, *a):
        refs = r.refs
        if f in refs.NoCache:
            s.core.cache.clear_cache()
        if f in refs.BuiltIns:
            return refs.BuiltIns[f](*a)
        # if f in refs.Functions:
        #     priority = {}
        #     for header in list(refs.Functions[f])[::-1]:
        #         match = True
        #         matches = 0
        #         if len(a) != len(header):
        #             continue
        #         for ar, br in zip(a, header):
        #             if ar == br:
        #                 matches += 1
        #                 continue
        #             elif isinstance(br, s.Symbol) and br.name.startswith('*'):
        #                 continue
        #             else:
        #                 match = False
        #                 break
        #         if match:
        #             priority[matches] = header
        #     if priority:
        #         header = priority[max(priority)]
        #         expr = refs.Functions[f][header][0]
        #         reps = refs.Functions[f][header][1]
        #         for ar, br in zip(a, header):
        #             if isinstance(br, s.Symbol) and br.name.startswith('*'):
        #                 expr = Subs(expr, Rule(br, ar))
        #         return Subs(expr, Rule.from_dict(vars(r.refs.Symbols)) + Rule.from_dict({x: x for x in reps}))
        return type(f, (NormalFunction,), {})(*a)

    @classmethod
    def pilot_call(cls, f: str, *a):
        return type(f, (PilotFunction,), {})(*a)

