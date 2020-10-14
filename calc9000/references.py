import sympy as s
from calc9000.custom import List, Primes


class NoOutput:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ''


class Protected:
    Symbol = None
    Rule = None
    List = None


Protected.Dict = {x: Protected.__dict__[x] for x in dir(Protected) if not x.startswith('__')}


class Constants:
    # Caveats: pi
    # TODO: Nothing and other constants
    # Progress: Elementary Functions, Numerical Functions
    Pi = s.pi
    pi = Pi
    I = s.I
    E = s.E
    EulerGamma = s.EulerGamma
    GoldenRatio = s.GoldenRatio
    Catalan = s.Catalan
    Infinity = s.oo
    ComplexInfinity = s.zoo
    Indeterminate = s.nan

    # sets
    Rationals = s.Rationals
    Complexes = s.Complexes
    Reals = s.Reals
    Integers = s.Integers
    Primes = Primes()

    Integer = None
    Real = None
    Head = None
    Complex = None
    All = s.Symbol('All')
    Nothing = s.Symbol('Nothing')
    Null = s.Symbol('Null')
    Degree = Pi / 180


setattr(Constants, 'True', True)
setattr(Constants, 'False', False)

Constants.Dict = {x: Constants.__dict__[x] for x in dir(Constants) if not x.startswith('__')}


class OwnValues(dict):
    pass


class DownValues(dict):
    pass


class TagValues(dict):
    pass


# Build from built-ins at startup
FunctionWrappers = {
    # Trig Functions
    'Sin': 'sin',
    'Cos': 'cos',
    'Tan': 'tan',
    'Csc': 'csc',
    'Sec': 'sec',
    'Cot': 'cot',
    'Sinh': 'sinh',
    'Cosh': 'cosh',
    'Tanh': 'tanh',
    'Csch': 'csch',
    'Sech': 'sech',
    'Coth': 'coth',
    'ArcSin': 'asin',
    'ArcCos': 'acos',
    'ArcTan': 'atan',
    'ArcCsc': 'acsc',
    'ArcSec': 'asec',
    'ArcCot': 'acot',
    'ArcSinh': 'asinh',
    'ArcCosh': 'acosh',
    'ArcTanh': 'atanh',
    'ArcCsch': 'acsch',
    'ArcSech': 'asech',
    'ArcCoth': 'acoth',
    'Log': 'log',
    'Floor': 'floor',
    'Ceiling': 'ceiling',
    'Re': 're',
    'Im': 'im',
    'Arg': 'arg',
    # 'Transpose': 'transpose',
    # 'Inverse': 'inverse',
    'Factorial': 'factorial',
    'Factorial2': 'factorial2',
    'Conjugate': 'conjugate',
    'Sqrt': 'sqrt',
    'StieltjesGamma': 'stieltjes',
    'Gamma': 'gamma',
    'PolyGamma': 'polygamma',
    'Surd': 'real_root',
    'GCD': 'gcd',
    'LCM': 'lcm',
    'PrimeQ': 'isprime',
    'Equal': 'eq',
    'Factor': 'factor',
    'Expand': 'expand',
    'TrigExpand': 'expand_trig',
    'HeavisideTheta': 'Heaviside',
    'Simplify': 'simplify',
    'Zeta': 'zeta',
    'LogIntegral': 'li',
    'ExpIntegralEi': 'Ei',
    'ExpIntegralE': 'expint',
    'SinIntegral': 'Si',
    'CosIntegral': 'Ci',
    'SinhIntegral': 'Shi',
    'CoshIntegral': 'Chi',
    'EllipticK': 'elliptic_k',
    'EllipticF': 'elliptic_f',
    'EllipticE': 'elliptic_e',
    'EllipticPi': 'elliptic_pi',
    'Erf': 'erf',
    'Erfc': 'erfc',
    'Erfi': 'erfi',
    'InverseErf': 'erfinv',
    'InverseErfc': 'erfcinv',
    'FresnelS': 'fresnels',
    'FresnelC': 'fresnelc'
}

FunctionWrappersReverse = {v: k for k, v in FunctionWrappers.items()}

FunctionWrappersReverse.update({
    'erf2': 'Erf',
})

NoCache = [
    "Out",
    "ReplaceAll",
    "Set",
    "SetDelayed",
    "Subs",
    "Unset",
    "Clear",
    "SeedRandom",
    "Random",
    "RandomInteger",
    "RandomReal",
    "RandomChoice",
    "RandomComplex",
    "RandomPrime",
    "ComplexExpression",
    "SemicolonStatement"
]


class BaseMessenger:
    def show(self, tag, e):
        pass


class References:
    def __init__(self):
        self.In = []
        self.Out = []
        self.Messages = [List()]
        self.OwnValues = OwnValues()
        self.NoCache = NoCache
        self.BuiltIns = DownValues()
        self.DownValues = DownValues()
        self.TagValues = TagValues()
        self.FunctionWrappers = FunctionWrappers
        self.Constants = Constants
        self.Protected = Protected
        self.Line = 1
        self.CacheClearQueued = False  # dirty, but works
        self.Parser = None
        self.Messenger = None
        self.DefaultPrecision = 9
        self.ExtraPrecision = 5
        self.WorkingPrecision = self.DefaultPrecision + self.ExtraPrecision

    def add_def(self, _in, out):
        self.In.append(_in)
        self.Out.append(out)
        self.Messages.append(List())
        self.Line += 1

    def get_def(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.In[n - 1], self.Out[n - 1]

    def get_in(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.In[n - 1]

    def get_out(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.Out[n - 1]

    def add_message(self, tag, e):  # WIP
        self.Messages[-1].append((tag, e))
        if self.Messenger:
            self.Messenger.show(tag, e)

    def get_messages(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.Messages[n - 1]


refs = References()
