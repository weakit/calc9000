import sympy as s
from calc9000.datatypes import List


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
    Complexes = s.Complexes
    Reals = s.Reals
    Integers = s.Integers
    Integer = None
    Real = None
    Head = None
    Complex = None
    Rationals = s.Rationals
    All = s.Symbol('All')
    Nothing = s.Symbol('Nothing')
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


FunctionWrappers = {
    # Trig Functions
    'Sin': 'sin',
    'Cos': 'cos',
    'Tan': 'tan',
    'Csc': 'csc',
    'Sec': 'sec',
    'Cot': 'cot',
    'ArcSin': 'asin',
    'ArcCos': 'acos',
    'ArcTan': 'atan',
    'ArcCsc': 'acsc',
    'ArcSec': 'asec',
    'ArcCot': 'acot',
    'Log': 'log',
    'Floor': 'floor',
    'Ceiling': 'ceiling',
    'Re': 're',
    'Im': 'im',
    'Arg': 'arg',
    # 'Transpose': 'transpose',
    # 'Inverse': 'inverse',
    'Factorial': 'factorial',
    'Conjugate': 'conjugate',
    'Sqrt': 'sqrt',
    'StieltjesGamma': 'stieltjes',
    'Gamma': 'gamma',
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
}

FunctionWrappersReverse = {v: k for k, v in FunctionWrappers.items()}

NoCache = [
    "Out",
    "ReplaceAll",
    "Set",
    "Subs",
    "Unset",
    "Random",
    "RandomInteger",
    "RandomReal",
    "RandomComplex",
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
        self.Messages[-1].append(f'{tag}: {e}')
        if self.Messenger:
            self.Messenger.show(tag, e)

    def get_messages(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.Messages[n - 1]


refs = References()
