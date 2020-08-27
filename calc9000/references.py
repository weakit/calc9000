import sympy as s


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


class Symbols(dict):
    pass


class Functions(dict):
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

NoCache = [
    "Out",
    "ReplaceAll",
    "Set",
    "Subs",
    "Unset",
    "Random",
    "RandomInteger",
    "RandomReal",
    "RandomComplex"
]


class References:
    def __init__(self):
        self.In = [None]
        self.Out = [None]
        self.Symbols = Symbols()
        self.NoCache = NoCache
        self.BuiltIns = Functions()
        self.Functions = Functions()
        self.FunctionWrappers = FunctionWrappers
        self.Constants = Constants
        self.Protected = Protected
        self.Line = 1
        self.Parser = None

    def add_def(self, _in, out):
        self.In.append(_in)
        self.Out.append(out)
        self.Line += 1

    def get_def(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.In[n], self.Out[n]

    def get_in(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.In[n]

    def get_out(self, n=None):
        if n is None:
            n = self.Line - 1
        return self.Out[n]


refs = References()
