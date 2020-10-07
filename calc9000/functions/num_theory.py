from calc9000.functions.core import *


class GCD(NormalFunction):
    """
    GCD [x1, x2, x3, …]
     Gives the GCD of x1, x2, x3, …

    Works with Numeric and Symbolic expressions.

    Equivalent to sympy.gcd()
    """

    @classmethod
    def exec(cls, *n):
        if len(n) == 1:
            return n[0]
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
    def exec(cls, *n):
        if len(n) == 1:
            return n[0]
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
    def exec(cls, n):
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
        return None

    @classmethod
    def exec(cls, n):
        return thread(cls._comp, n)


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

    Effectively uses sympy.factorint()
    """
    @staticmethod
    def prime_func(n):
        return sum(s.factorint(n).values())


class PrimeNu(Prime):
    """
    PrimeNu [x]
     Gives the number of distinct primes in x.

    Effectively uses sympy.primefactors()
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

    Effectively uses sympy.randprime().
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
