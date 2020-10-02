from calc9000.functions.core import *
from calc9000.functions.base import Part
from iteration_utilities import deepflatten, accumulate
from itertools import permutations, combinations


class Min(NormalFunction):
    """
    Min [x1, {x2, x3}, x4, …]
     Gives the smallest x.
    """

    @classmethod
    def exec(cls, *x):
        x = deepflatten(x)
        return s.Min(*x)


class Max(NormalFunction):
    """
    Max [x1, {x2, x3}, x4, …]
     Gives the largest x.
    """

    @classmethod
    def exec(cls, *x):
        x = deepflatten(x)
        return s.Max(*x)


class Total(NormalFunction):
    """
    Total [list]
     Gives the Total Sum of elements in list.
    """

    @classmethod
    def exec(cls, _list):
        if isinstance(_list, iterables):
            return sum(_list)
        return None


class Mean(NormalFunction):
    """
    Mean [list]
        Gives the statistical mean of elements in list.
    """

    @classmethod
    def exec(cls, _list):
        if isinstance(_list, iterables):
            return Total(_list) / len(_list)
        return None


class Accumulate(NormalFunction):
    """
    Accumulate [list]
     Gives a list of the successive accumulated totals of elements in list.
    """

    @classmethod
    def exec(cls, expr):
        if isinstance(expr, iterables):
            head = List
            iterable = expr
        else:
            head = expr.__class__
            iterable = expr.args
        return head(*accumulate(iterable))


class Range(NormalFunction):
    """
    Range [i]
     Generates the list {1, 2, …, i}.

    Range [a, b]
     Generates the list {a, …, b}.

    Range[a, b, di]
     Uses step di.
    """

    tags = {
        'range': 'Invalid range specification.'
    }

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
                raise FunctionException('Range::range')
        return List.create(ret)

    @classmethod
    def exec(cls, i, n=None, di=1):
        if any(isinstance(x, iterables) for x in (i, n, di)):
            return thread(cls.exec, i, n, di)
        return cls.single_range(i, n, di)


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
    def exec(cls, li, n=None):
        if n is not None:
            if isinstance(n, iterables):
                n = Range(*n)
            else:
                if not is_integer(n):
                    raise FunctionException('Permutations::exec', 'n should be an integer.')
                n = List.create(range(int(n) + 1))
        if isinstance(n, iterables):
            # TODO: manually remove duplicates
            ret = List()
            for i in n:
                ret = List.concat(ret, List.create(List.create(x) for x in set(permutations(li, int(i)))))
            return ret
        return List.create(List.create(x) for x in set(permutations(li, n)))


class Table(ExplicitFunction):
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
            li.append(LazyFunction.evaluate(Subs(expr, Rule(repl, arg))))
        return li

    @staticmethod
    def _range_parse(expr, arg):
        if hasattr(arg, '__len__') and len(arg) == 1:
            arg = arg[0]
        if arg.is_number:
            return List(*(LazyFunction.evaluate(expr) for _ in range(arg)))
        if len(arg) == 2 and isinstance(arg[1], iterables):
            args = arg[1]
        elif len(arg) >= 2:
            args = Range(*arg[1:])
        else:
            raise FunctionException('Table::range', 'Invalid Bounds.')  # TODO: Warning
        if not isinstance(arg[0], (s.Symbol, s.Function)):
            raise FunctionException('Table::iter', f'Cannot use {arg[0]} as an Iterator.')
        return Table._table(expr, arg[0], args)

    @classmethod
    def exec(cls, expr, *args):
        if not args:
            return LazyFunction.evaluate(expr)

        args = LazyFunction.evaluate(args)

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

    Subdivide [x, n]
     Generates the list of values obtained by subdividing the interval 0 to x.
     into n equal parts.

    Subdivide [min, max, n]
     Generates the list of values from subdividing the interval min to max.
    """

    tags = {
        'div': 'Number of Subdivisions should be an Integer.'
    }

    @classmethod
    def exec(cls, one, two=None, three=None):
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
            raise FunctionException('Subdivide::div')

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
    def exec(cls, li, n_spec=None):
        subsets = List()

        if n_spec is None:
            n_spec = range(len(li) + 1)
        elif n_spec.is_number:
            if not is_integer(n_spec):
                raise FunctionException('Subsets::exec', f'{n_spec} is not an integer.')
            n_spec = range(int(n_spec) + 1)
        else:
            n_spec = Range(*n_spec)

        for spec in n_spec:
            subsets.append(*(List.create(x) for x in combinations(li, spec)))

        return subsets


class Length(NormalFunction):
    """
    Length [expr]
     Gives the number of elements in expr.
    """

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            return len(x)
        if hasattr(x, 'args'):
            return len(x.args)
        return 0


class First(ExplicitFunction):
    """
    First [expr]
     Gives the first element in expr.

    First [expr, def]
     Gives the first element if it exists, or def otherwise.
    """

    @classmethod
    def exec(cls, x, d=None):
        x = LazyFunction.evaluate(x)
        if Length(x) > 0:
            return Part(x, 1)
        if d is not None:
            return LazyFunction.evaluate(d)
        raise FunctionException('First::first', f'{x} has zero length, and no first element.')


class Last(ExplicitFunction):
    """
    Last [expr]
     Gives the last element in expr.

    Last [expr, def]
     Gives the last element if there are any elements, or def otherwise.
    """

    @classmethod
    def exec(cls, x, d=None):
        x = LazyFunction.evaluate(x)
        if Length(x) > 0:
            return Part(x, -1)
        if d is not None:
            return LazyFunction.evaluate(d)
        raise FunctionException('Last::last', f'{x} has zero length, and no last element.')


class Prime(NormalFunction):
    """
    Prime [n]
     Gives the nth prime number.

    Equivalent to sympy.prime()
    """

    param_spec = (1, 1)
    tags = {
        'int': 'A positive integer is expected as input.'
    }

    prime_func = s.prime

    @classmethod
    def prime(cls, n):
        if is_integer(n):
            if n < 1:
                raise FunctionException(f'{cls.__name__}::int')
            return cls.prime_func(n)
        if hasattr(n, 'is_number') and n.is_number:
            raise FunctionException(f'{cls.__name__}::int')
        return None

    @classmethod
    def exec(cls, n):
        return thread(cls.prime, n)


class PrimePi(Prime):
    """
    PrimePi [n]
     Gives the number of primes less than or equal to x.

    Equivalent to sympy.primepi()
    """
    prime_func = s.primepi


class PrimeOmega(Prime):
    """
    PrimeOmega [x]
     Gives the number of prime factors counting multiplicities in x.

    Uses sympy.factorint()
    """
    @staticmethod
    def prime_func(n):
        return sum(s.factorint(n).values())


class PrimeNu(Prime):
    """
    PrimeNu [x]
     Gives the number of distinct primes in x.

    Uses sympy.primefactors()
    """
    @staticmethod
    def prime_func(n):
        return len(s.primefactors(n))


class NextPrime(NormalFunction):
    """
    NextPrime [x]
     Gives the smallest prime above x.

    NextPrime [x, i]
     Gives the ith-next prime above x.

    """

    tags = {
        'int': 'A positive integer is expected as input.'
    }

    prime_func = s.nextprime

    @classmethod
    def prime(cls, n, k):
        if is_integer(n) and is_integer(k):
            if n < 1:
                raise FunctionException(f'{cls.__name__}::int')
            return cls.prime_func(n, k)
        if hasattr(n, 'is_number') and n.is_number \
                or (hasattr(k, 'is_number') and k.is_number):
            raise FunctionException(f'{cls.__name__}::int')
        return None

    @classmethod
    def exec(cls, n, k=1):
        return thread(cls.prime, n, k)


class PreviousPrime(Prime):
    """
    PreviousPrime [x]
     Gives the greatest prime below x.
    """
    prime_func = s.prevprime


class RandomPrime(NormalFunction):
    """
    RandomPrime [{a, b}]
     Gives a pseudorandom prime number in the range a to b.

    RandomPrime [x]
     Gives a pseudorandom prime number in the range 2 to x.

    RandomPrime [range, n]
     Gives a list of n pseudorandom primes.

    Uses sympy.randprime().
    """

    tags = {
        'int': 'A positive integer is expected as input.'
    }

    @classmethod
    def exec(cls, spec, n=None):
        if not isinstance(spec, iterables):
            spec = List(2, spec)
        if not is_integer(spec[0]) and not is_integer(spec[1]):
            FunctionException(f'RandomPrime::int')

        if n:
            if is_integer(n):
                return List(*(cls.exec(spec) for _ in range(n)))
            if isinstance(n, iterables):
                return List(*(cls.exec(spec, n[1:]) for _ in range(n[0])))
        return s.randprime(spec[0], spec[1])


class Mobius(Prime):  # for ease of use
    """
    Mobius [n]
     Gives the Möbius function μ(n).
    """
    prime_func = s.mobius


class MoebiusMu(Prime):
    """
    MoebiusMu [n]
     Gives the Möbius function μ(n).
    """
    prime_func = s.mobius


class FactorInteger(NormalFunction):
    """
    FactorInteger [n]
     Gives a list of the prime factors of the integer n, together with their exponents.

    FactorInteger [n, k]
     Foes partial factorization, pulling out at most k distinct factors.

    Equivalent to sympy.factorint()
    """
    tags = {
        'int': 'Inputs should be positive integers.'
    }

    @classmethod
    def factor(cls, n, k):
        if is_integer(k) or k is None:
            if hasattr(n, 'is_Rational') and n.is_Rational:
                return List.create(List(*x) for x in s.factorrat(n, limit=k).items())
        if hasattr(n, 'is_number') and n.is_number \
                or (hasattr(k, 'is_number') and k.is_number):
            raise FunctionException(f'{cls.__name__}::int')
        return None

    @classmethod
    def exec(cls, n, k=None):
        return thread(cls.factor, n, k)


class Divisible(NormalFunction):
    tags = {
        'rat': 'Rational numbers.py are expected as input.'
    }

    @staticmethod
    def div(n, d):
        if n.is_number and d.is_number:
            if n.is_real and d.is_real:
                if n.is_Float or d.is_Float:  # make sure input is symbolic
                    raise FunctionException('Divisible::rat')
                return not bool(s.Mod(n, d))
            else:
                div = n / d
                if s.re(div).is_Rational and s.im(div).is_Rational:
                    return s.re(div).is_Integer and s.im(div).is_Integer
                raise FunctionException('Divisible::rat')
        return None

    @classmethod
    def exec(cls, n, k):
        if isinstance(n, iterables) or isinstance(k, iterables):
            return thread(Divisible, n, k)
        return cls.div(n, k)
