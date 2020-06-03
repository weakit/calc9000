import sympy as s


class Protected:
    Symbol = None
    Rule = None
    List = None


class Constants:
    # Caveats: pi
    # TODO: Degrees, Nothing and other constants
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
    Rationals = s.Rationals
    All = s.Symbol('All')


class Symbols:
    pass


class Functions:
    pass


NoCache = [
    "Out",
    "ReplaceAll",
    "Set",
    "Subs",
    "Unset"
]


class References:
    def __init__(self):
        self.In = [None]
        self.Out = [None]
        self.Symbols = Symbols()
        self.NoCache = NoCache
        self.Functions = Functions()
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
