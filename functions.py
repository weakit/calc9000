import operator
from functools import reduce
import references as r
import sympy as s
from sympy.printing.pretty.stringpict import stringPict, prettyForm, xsym
from collections.abc import Iterable, Sized
from lists import List, Rule


class FunctionException(Exception): pass


# TODO: add warnings
def toList(m):
    temp_list = list()
    for row in range(m.rows):
        temp_list.append(List(m.row(row)))
    return List(temp_list)


def thread(x, func):
    if isinstance(x, s.Matrix):
        x = toList(x)
    if isinstance(x, s.Tuple):
        temp_list = list()
        for item in x:
            temp_list.append(thread(item, func))
        return List(temp_list)
    return func(x)


def assumptions(x):
    a = True
    for assumption in x:
        a = a & assumption
    return a


class Exp(s.Function):
    @classmethod
    def eval(cls, z):
        return thread(z, lambda z: pow(s.E, z))


class Log(s.Function):
    @classmethod
    def eval(cls, x, b=None):
        if b is not None:
            return thread(x, lambda a: s.log(b, a))
        return thread(x, s.log)


class Log2(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, lambda a: s.log(a, 2))


class Log10(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, lambda a: s.log(a, 10))


class Round(s.Function):
    @classmethod
    def eval(cls, x, a=None):
        if x.is_number:
            if a is None:
                return round(x)
            return a * round(x / a)
        if isinstance(x, (s.Tuple, s.Matrix)):
            return thread(x, Round)


class Floor(s.Function):
    @classmethod
    def eval(cls, x, a=None):
        if a is None:
            return thread(x, s.floor)
        return thread(x, lambda y: a * s.floor(y / a))


class Ceiling(s.Function):
    @classmethod
    def eval(cls, x, a=None):
        if a is None:
            return thread(x, s.ceiling)
        return thread(x, lambda y: a * s.ceiling(y / a))


def Min(*x):
    temp_list = list()
    for i in x:
        if isinstance(i, (Iterable, s.Matrix, s.Tuple)):
            temp_list.append(Min(*i))
        else:
            temp_list.append(i)
    return s.Min(*temp_list)


def Max(*x):
    temp_list = list()
    for i in x:
        if isinstance(i, (Iterable, s.Matrix, s.Tuple)):
            temp_list.append(Max(*i))
        else:
            temp_list.append(i)
    return s.Max(*temp_list)


class Total(s.Function):
    @classmethod
    def eval(cls, _list):
        if isinstance(_list, s.Tuple):
            return sum(_list)


class Mean(s.Function):
    @classmethod
    def eval(cls, _list):
        if isinstance(_list, s.Tuple):
            return Total(_list) / len(_list)


class Accumulate(s.Function):
    @classmethod
    def eval(cls, _list):
        temp_list = list(_list)
        if isinstance(_list, s.Tuple):
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
        if x_range is None and isinstance(x, s.Tuple):
            x = list(x)
            _min = Min(x)
            _max = Max(x)
            for i in range(len(x)):
                x[i] = cls.eval(x[i], List([_min, _max]))
            return List(x)
        elif isinstance(x_range, s.Tuple) and len(x_range) == 2:
            if y_range is None or (isinstance(y_range, List) and len(y_range) == 2):
                if y_range is None:
                    y_range = (0, 1)
                return ((x - x_range[0]) * y_range[1] + (x_range[1] - x) * y_range[0]) / (x_range[1] - x_range[0])


class In(s.Function):
    @classmethod
    def eval(cls, n=None):
        if n is None:
            return r.refs.get_in()
        if n.is_Integer and 0 < n < r.refs.Line:
            return r.refs.get_in(n)


class Out(s.Function):
    @classmethod
    def eval(cls, n=None):
        out = None
        if n is None:
            out = r.refs.get_out()
        if isinstance(n, (s.Number, int, float)) and 0 < n < r.refs.Line:
            out = r.refs.get_out(n)
        if isinstance(out, s.Expr):  # TODO: Replace with Subs func.
            out = out.subs(vars(r.refs.Symbols))
        return out


class Dot(s.Function):
    @classmethod
    def eval(cls, m, n):
        if not ((isinstance(m, s.Tuple) or isinstance(m, s.Matrix)) and (isinstance(n, s.Tuple) or isinstance(n, s.Matrix))):
            return None
        m = s.Matrix(m)
        n = s.Matrix(n)

        if m.shape[1] == n.shape[1] == 1:
            return m.dot(n)
        return m * n

    def _pretty(self, printer=None):
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
    @classmethod
    def eval(cls, x):
        if isinstance(x, (s.Tuple, s.Matrix)):
            try:
                m = s.Matrix(x)
                if m.is_square:
                    return m.det()
            except ValueError:
                return None


class Inverse(s.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, (s.Tuple, s.Matrix)):
            try:
                m = s.Matrix(x)
                return toList(m.inv())
            except ValueError:
                return None


class Transpose(s.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, (s.Tuple, s.Matrix)):
            try:
                m = s.Matrix(x)
                return toList(m.transpose())
            except ValueError:
                return None


class Re(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.re)


class ReIm(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, lambda b: List((s.re(b), s.im(b))))


class Im(s.Function):
    @classmethod
    def eval(cls, x):
        thread(x, s.im)


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
    @classmethod
    def eval(cls, x):
        return thread(x, s.Abs)


class Arg(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.arg)


class AbsArg(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, lambda y: List((Abs(y), Arg(y))))


class Factorial(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.factorial)


class Conjugate(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.conjugate)


class ConjugateTranspose(s.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, (s.Tuple, s.Matrix)):
            return Transpose(Conjugate(x))


class ComplexExpand(s.Function):
    @classmethod
    def eval(cls, x, complexes=tuple()):
        def exp(expr):
            return s.refine(s.expand_complex(expr),
                            assumptions(s.Q.real(a) for a in expr.atoms(s.Symbol) if a not in complexes))
        if not isinstance(complexes, (tuple, s.Tuple)):
            complexes = (complexes,)
        return thread(x, exp)


class LogisticSigmoid(s.Function):
    @classmethod
    def eval(cls, z):
        return thread(z, lambda x: 1 / (1 + s.exp(-x)))


class Unitize(s.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, (tuple, s.Tuple, s.Matrix)):
            return thread(x, Unitize)
        if s.ask(s.Q.nonzero(x)):
            return


class Ramp(s.Function):
    @classmethod
    def eval(cls, x):
        if isinstance(x, (tuple, s.Tuple, s.Matrix)):
            return thread(x, Ramp)
        if s.ask(s.Q.nonnegative(x)):
            return x
        return 0


class Cross(s.Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], (s.Tuple, s.Matrix)) and len(args[0]) == 2:
                return List((args[0][1] * -1, args[0][0]))
        elif len(args) == 2:
            if isinstance(args[0], (s.Tuple, s.Matrix)) and isinstance(args[1], (s.Tuple, s.Matrix)):
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
    @classmethod
    def eval(cls, x):
        return thread(x, s.sign)


class Series(s.Function):
    # TODO: Series function
    @classmethod
    def eval(cls, f, *x):
        pass


class Sqrt(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.sqrt)


class StieltjesGamma(s.Function):
    @classmethod
    def eval(cls, x):
        return thread(x, s.stieltjes)


class Surd(s.Function):
    @classmethod
    def eval(cls, x, n):
        return thread(x, lambda a: s.real_root(a, n))


class QuotientRemainder(s.Function):
    @classmethod
    def eval(cls, m, n):
        if m.is_number and n.is_number:
            return List((m // n, m % n))
        if isinstance(m, (tuple, s.Tuple, List)) and isinstance(n, (tuple, s.Tuple, List)):
            return List(QuotientRemainder(*x) for x in zip(m, n))


class GCD(s.Function):
    @classmethod
    def eval(cls, *n):
        gcd = n[0]
        for number in n[1:]:
            gcd = s.gcd(gcd, number)
        return gcd


class LCM(s.Function):
    @classmethod
    def eval(cls, *n):
        lcm = n[0]
        for number in n[1:]:
            lcm = s.lcm(lcm, number)
        return lcm


class PrimeQ(s.Function):
    @classmethod
    def eval(cls, n):
        return thread(n, s.isprime)


class CompositeQ(s.Function):
    @classmethod
    def eval(cls, n):
        def comp(x):
            if x.is_number:
                if x.is_composite:
                    return True
                return False
        return thread(n, comp)


class Equal(s.Function):
    @classmethod
    def eval(cls, *args):
        if len(args) == 2:
            return s.Eq(args[0], args[1])
        # TODO: better multiple equality
        return s.And(*[s.Eq(args[x], args[x + 1]) for x in range(len(args) - 1)])


class Set(s.Function):
    @classmethod
    def eval(cls, n, x):
        # TODO: Function Assignment
        for ref in [r.refs.Constants, r.refs.Functions]:
            if str(n) in ref.__dict__:
                # TODO: warning
                return None
        if isinstance(n, s.Symbol):
            if isinstance(x, s.Expr):
                if n in x.atoms():
                    return None
            r.refs.Symbols.__setattr__(n.name, x)
            return x
        elif isinstance(n, (s.Tuple, Sized)):
            if isinstance(n, (s.Tuple, Sized)) and len(n) == len(x):
                return List(Set(a, b) for a, b in zip(n, x))


class Unset(s.Function):
    @classmethod
    def eval(cls, n):
        if isinstance(n, (s.Tuple, Sized)):
            return List(Unset(x) for x in n)
        if isinstance(n, s.Symbol):
            delattr(r.refs.Symbols, str(n))
            return None
        delattr(r.refs.Symbols, str(n))
        return None


class Rationalize(s.Function):
    @classmethod
    def eval(cls, x, dx=None):
        if isinstance(x, (s.Tuple, Sized)):
            return List(Rationalize(n, dx) for n in x)
        if isinstance(x, (int, float, s.Number)):
            rat = s.nsimplify(x, rational=True, tolerance=dx)
            if 1 / (10**4 * s.denom(rat)) > s.Abs(rat - x):
                return rat
        return x


class Subs(s.Function):
    @classmethod
    def eval(cls, expr, *replacements):
        if isinstance(expr, (s.Tuple, List)):
            return List(Subs(x, *replacements) for x in expr)
        if isinstance(expr, s.Expr):
            expr = expr.Subs(replacements)
            replacement_dict = {str(k): str(v) for k, v in replacements}
            for func in expr.atoms(s.Function):
                if func.name in replacement_dict:
                    expr = expr.replace(func, Functions.call(replacement_dict[func.name], *func.args))
            return expr.Subs(replacements)


class ReplaceAll(s.Function):
    @classmethod
    def eval(cls, expr, rules):
        if isinstance(rules, Rule):
            return Subs(expr, rules)
        return Subs(expr, *rules)


class Factor(s.Function):
    @classmethod
    def eval(cls, expr):
        return thread(expr, s.factor)


class nPr(s.Function):
    @classmethod
    def eval(cls, n, l):
        def npr(x, q):
            return Factorial(x) / Factorial(x - q)
        return thread(n, lambda a: npr(a, l))


class nCr(s.Function):
    @classmethod
    def eval(cls, n, l):
        def ncr(x, q):
            return Factorial(x) / (Factorial(x - q) * Factorial(q))
        return thread(n, lambda a: ncr(a, l))


class Functions:
    # TODO: Solve
    # TODO: D, Integrate
    # TODO: Simplify, Expand
    # TODO: Fractional, Integer Part
    # TODO: Remaining Matrix Operations
    # TODO: Arithmetic Functions: Ratios, Differences (Low Priority)
    # TODO: Booleans, Conditions, Boole (Low Priority)
    # TODO: Cross of > 3 dimensional vectors (Low Priority)
    # TODO: Implement Fully: Total, Clip, Quotient, Mod, Factor (Low Priority)

    Abs = Abs
    AbsArg = AbsArg
    Arg = Arg
    Accumulate = Accumulate
    Clip = Clip
    Ceiling = Ceiling
    ComplexExpand = ComplexExpand
    CompositeQ = CompositeQ
    Conjugate = Conjugate
    ConjugateTranspose = ConjugateTranspose
    Cross = Cross
    D = s.diff
    Det = Det
    Dot = Dot
    Equal = Equal
    Exp = Exp
    Factor = Factor
    Factorial = Factorial
    Floor = Floor
    GCD = GCD
    In = In
    Im = Im
    Inverse = Inverse
    LCM = LCM
    Log = Log
    Log10 = Log10
    Log2 = Log2
    LogisticSigmoid = LogisticSigmoid
    Max = Max
    Mean = Mean
    Min = Min
    Mod = s.Mod
    N = s.N
    Out = Out
    Plus = Plus
    Power = Power
    PowerMod = PowerMod
    PrimeQ = PrimeQ
    Quotient = Quotient
    QuotientRemainder = QuotientRemainder
    Ramp = Ramp
    Rationalize = Rationalize
    Re = Re
    ReIm = ReIm
    ReplaceAll = ReplaceAll
    Rescale = Rescale
    Round = Round
    Set = Set
    Series = Series
    Sign = Sign
    Simplify = s.simplify
    Sqrt = Sqrt
    StieltjesGamma = StieltjesGamma
    Subtract = Subtract
    Subs = Subs
    Surd = Surd
    Times = Times
    Total = Total
    Unitize = Unitize
    Unset = Unset

    # Trig Functions

    Sinc = s.sinc
    Sin = s.sin
    Cos = s.cos
    Tan = s.tan
    Csc = s.csc
    Sec = s.sec
    Cot = s.cot
    Sinh = s.sinh
    Cosh = s.cosh
    Tanh = s.tanh
    Csch = s.csch
    Sech = s.sech
    Coth = s.coth
    ArcSin = s.asin
    ArcCos = s.acos
    ArcTan = s.atan
    ArcCsc = s.acsc
    ArcSec = s.asec
    ArcCot = s.acot
    ArcSinh = s.asinh
    ArcCosh = s.acosh
    ArcTanh = s.atanh
    ArcCsch = s.acsch
    ArcSech = s.asech
    ArcCoth = s.acoth

    # Extra functions

    nCr = nCr
    NCr = nCr
    nPr = nPr
    NPr = nPr

    @classmethod
    def call(cls, f, *a):
        if f in r.refs.NoCache:
            s.cache.clear_cache()
        if f in cls.__dict__:
            if a:
                return cls.__dict__[f](*a)
            return cls.__dict__[f]
        if a:
            return s.Function(f)(*a)
        return s.Function(f)
