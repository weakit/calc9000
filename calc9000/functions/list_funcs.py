from bisect import bisect_left
from itertools import combinations, islice, permutations

from iteration_utilities import (accumulate, deepflatten, nth_combination,
                                 unique_everseen)

from calc9000.functions.core import *


def list_to_python_list(n):
    if isinstance(n, List):
        return [list_to_python_list(x) for x in n.value]
    return n


def do_list_add_mul(n):  # solve dilemma
    if isinstance(n, iterables):
        return List(*(do_list_add_mul(x) for x in n))
    if isinstance(n, s.Add):
        return Plus.exec(*(do_list_add_mul(x) for x in n.args))
    if isinstance(n, s.Mul):
        return Times.exec(*(do_list_add_mul(x) for x in n.args))
    return n


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
                raise FunctionException(
                    "Part::dim", f"Part {n} of {expr} does not exist."
                )
        raise FunctionException("Part::part", f"{n} is not a valid Part specification.")

    @classmethod
    def exec(cls, expr, *args):
        part = head = None

        if not args:
            return expr

        if hasattr(expr, "args"):
            part = expr.args
            head = expr.__class__

        elif hasattr(expr, "__getitem__"):
            part = expr
            head = List

        arg = args[0]

        if arg == s.S.Zero:
            return s.Symbol(type(expr).__name__)

        if not part:
            raise FunctionException("Part::dim", f"{expr} does not have Part {arg}")

        if arg == r.refs.Constants.All:  # TODO: add None
            arg = Span()

        if isinstance(arg, Span):
            return List(
                *(Part(x, *args[1:]) for x in Take(expr, arg))
            )  # pass expr with head

        if isinstance(arg, iterables):
            return head(*(Part(cls.get_part(part, x), *args[1:]) for x in arg))

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
    def get_take(cls, take, seq):
        if seq == r.refs.Constants.All:
            return take
        # TODO: add None
        if isinstance(seq, Span):
            seq = seq.take_spec()
        if isinstance(seq, iterables):
            if len(seq) == 1:
                return Part(take, seq)
            lower = seq[0]
            upper = seq[1]
            step = 1
            if len(seq) == 3:
                step = seq[2]
            if (
                0 in (lower, upper, step)
                or not (is_integer(lower) and is_integer(upper) and is_integer(step))
            ) or len(seq) > 3:
                raise FunctionException("Take::dim", "Invalid Bounds for Take.")
            if step > 0:
                upper, lower = cls.ul(upper, lower)
            else:
                upper -= 1
                lower, upper = cls.ul(lower, upper)
            return take[lower:upper:step]
        if is_integer(seq):
            if seq > 0:
                return take[:seq]
            return take[seq:]
        raise FunctionException(
            "Take::take", f"{seq} is not a valid Take specification."
        )

    @classmethod
    def exec(cls, expr, *seqs):
        take = head = None

        if not seqs:
            return expr
        if hasattr(expr, "args"):
            take = expr.args
            head = expr.__class__
        elif hasattr(expr, "__getitem__"):
            take = expr
            head = List

        if len(seqs) > 1:
            return head(*(cls.exec(x, *seqs[1:]) for x in cls.get_take(take, seqs[0])))
        return head(*cls.get_take(take, seqs[0]))


class Min(NormalFunction):
    """
    Min [x1, {x2, x3}, x4, …]
     Gives the smallest x.
    """

    @classmethod
    def exec(cls, *x):
        try:  # for better performance
            return min(*deepflatten(x))
        except TypeError:
            return s.Min(*deepflatten(x))


class Max(NormalFunction):
    """
    Max [x1, {x2, x3}, x4, …]
     Gives the largest x.
    """

    @classmethod
    def exec(cls, *x):
        try:
            return max(*deepflatten(x))
        except TypeError:
            return s.Max(*deepflatten(x))


class Total(NormalFunction):
    """
    Total [list]
     Gives the Total Sum of elements in list.

    Total [list, n]
     Totals all elements down to level n.

    Total [list, {n}]
     Totals elements at level n.

    Total [list, {a, b}]
     Totals elements at levels a through b.

    Negative levels are not supported.
    Effectively uses sum().
    """

    # TODO: Proper Levels (including negative)
    # TODO: Total: AllowedHeads

    tags = {"level": "A positive integer is expected as a level."}

    @classmethod
    def total(cls, m, level, levels, max_level):
        if not isinstance(m, iterables):
            return m
        if level > max_level:
            return m

        if level in levels:
            return sum(cls.total(x, level + 1, levels, max_level) for x in m)
        return List(*(cls.total(x, level + 1, levels, max_level) for x in m))

    @classmethod
    def exec(cls, x, levels=List(s.S.One)):
        if not isinstance(levels, iterables):
            if levels.is_number and levels.is_integer and levels > 0:
                levels = Range(levels)
            else:
                raise FunctionException("Total::level")
        else:
            if not all(x.is_number and x.is_integer and x > 0 for x in levels):
                raise FunctionException("Total::level")

        return cls.total(x, 1, levels, Max(levels))


class Mean(NormalFunction):
    """
    Mean [list]
        Gives the statistical mean of elements in list.
    """

    @classmethod
    def exec(cls, _list):
        if isinstance(_list, iterables):
            return sum(_list) / len(_list)
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

    tags = {"range": "Invalid range specification."}

    @staticmethod
    def single_range(i, n, di):
        if n is None:
            n = i
            i = s.S.One

        iters = (n - i) / di

        if not iters.is_number:
            # try simplifying in case bounds are symbolic
            iters = s.simplify(iters)
            if not iters.is_number:  # raise if can't iterate
                raise FunctionException("Range::range")
        if not iters >= 0:
            raise FunctionException("Range::range")

        return List(*(i + x * di for x in range(int(iters) + 1)))

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

    Effectively uses itertools.permutations().
    """

    tags = {"int": "", "num": "A non-negative Integer is expected as a specification."}

    @classmethod
    def permute(cls, obj, perms):
        ret = []

        if isinstance(obj, iterables):
            head = List
            it = obj
        else:
            head = obj.__class__
            it = obj.args

        for per in perms:
            if not is_integer(per):
                raise FunctionException("Permutations::int")
            ret += [head(*x) for x in permutations(it, int(per))]

        return List(*unique_everseen(ret))

    @classmethod
    def exec(cls, li, n=None):
        if not isinstance(n, iterables):
            if not n:
                n = (Length(li),)
            else:
                if n.is_number:
                    n = Range(0, n)
                elif n is r.Constants.All:
                    n = Range(0, Length(li))
                else:
                    raise FunctionException("Permutations::num")
        else:
            if len(n) > 1:
                n = Range(*n)

        return cls.permute(li, n)


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
        li = []
        for arg in args:
            li.append(LazyFunction.evaluate(Replace(expr, Rule(repl, arg))))
        return List.create(li)

    @staticmethod
    def _range_parse(expr, arg):
        if hasattr(arg, "__len__") and len(arg) == 1:
            arg = arg[0]
        if arg.is_number:
            return List(*(LazyFunction.evaluate(expr) for _ in range(arg)))
        if len(arg) == 2 and isinstance(arg[1], iterables):
            args = arg[1]
        elif len(arg) >= 2:
            args = Range(*arg[1:])
        else:
            raise FunctionException("Table::range", "Invalid Bounds.")  # TODO: Warning
        if not isinstance(arg[0], (s.Symbol, s.Function)):
            raise FunctionException(
                "Table::iter", f"Cannot use {arg[0]} as an Iterator."
            )
        return Table._table(expr, arg[0], args)

    @classmethod
    def exec(cls, expr, *args):
        if not args:
            return LazyFunction.evaluate(expr)

        args = LazyFunction.evaluate(args)

        if len(args) == 1:
            return cls._range_parse(expr, args[0])

        li = []
        for expr_, specs in zip(
            cls._range_parse(expr, args[0]), cls._range_parse(args[1:], args[0])
        ):
            li.append(Table(expr_, *specs))

        return List(*li)


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

    tags = {"div": "Number of Subdivisions should be an Integer."}

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
            raise FunctionException("Subdivide::div")

        div = s.Number(int(div))

        step = (x_max - x_min) / div
        li = [x_min]

        for _ in range(int(div)):
            li.append(li[-1] + step)
        return List(*li)


class Subsets(NormalFunction):
    """
    Subsets [list]
     Gives a list of all possible subsets of list. (Power Set)

    Subsets [list, n]
     Gives all subsets containing at most n elements.

    Subsets [list, {n}]
     Gives all subsets containing exactly n elements.

    Subsets[list, {a, b}]
     Gives all subsets containing between a and b elements.

    Subsets[list, spec, s]
     Limits the result to the first s subsets.

    Effectively uses itertools.combinations().
    """

    # TODO: Treat different occurrences of same element as distinct
    # TODO: Better error messages.

    tags = {
        "int": "",
        "num": "A non-negative Integer is expected as a specification.",
        "lim": "An invalid limit was encountered.",
    }

    @classmethod
    def ncr(cls, a, b):
        return s.factorial(a) / (s.factorial(a - b) * s.factorial(b))

    @classmethod
    def nth_combine(cls, obj, perms, ns):
        _len = len(obj)
        crs = tuple(accumulate([cls.ncr(_len, x) for x in perms]))
        ret = []

        try:
            for n in ns:
                b = bisect_left(crs, n)
                ret.append(List(*nth_combination(obj, perms[b], n - crs[b] - 1)))
        except (ValueError, IndexError):
            raise FunctionException("Subsets::lim")

        return List(*ret)

    @classmethod
    def combine(cls, obj, perms, limit=None):
        ret = []
        limit = limit and int(limit)

        if isinstance(obj, iterables):
            head = List
            it = obj
        else:
            head = obj.__class__
            it = obj.args

        if limit:
            for per in perms:
                if not is_integer(per):
                    raise FunctionException("Subsets::int")
                ret += [
                    head(*x)
                    for x in islice(combinations(it, int(per)), limit - len(ret))
                ]

                if len(ret) >= limit:
                    break
        else:
            for per in perms:
                if not is_integer(per):
                    raise FunctionException("Subsets::int")
                ret += [head(*x) for x in combinations(it, int(per))]

        return List(*unique_everseen(ret))

    @classmethod
    def exec(cls, li, n=None, limit=None):
        if not isinstance(n, iterables):
            if not n:
                n = Range(0, Length(li))
            else:
                if n.is_number:
                    n = Range(0, n)
                elif n is r.Constants.All:
                    n = Range(0, Length(li))
                else:
                    raise FunctionException("Subsets::num")
        else:
            if len(n) > 1:
                n = Range(*n)

        if limit:
            if isinstance(limit, iterables):
                if len(limit) > 1:
                    limit = Range(*limit)
                return cls.nth_combine(li, n, limit)
            if limit.is_number and limit.is_Integer:
                limit = int(limit)
            else:
                raise FunctionException("Subsets::lim")

        return cls.combine(li, n, limit)


class Length(NormalFunction):
    """
    Length [expr]
     Gives the number of elements in expr.
    """

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            return len(x)
        if hasattr(x, "args"):
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
        raise FunctionException(
            "First::first", f"{x} has zero length, and no first element."
        )


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
        raise FunctionException(
            "Last::last", f"{x} has zero length, and no last element."
        )


class Reverse(NormalFunction):
    """
    Reverse [expr]
     Reverses the order of the elements in expr.

    Reverse [expr, n]
     Reverses elements at level n in expr.

    Reverse [expr, {a, b, …}]
     Reverses elements at levels a, b, … in expr.
    """

    tags = {"level": "A positive integer is expected as a level."}

    @classmethod
    def reverse(cls, m, level, levels, max_level):
        # only works with Lists and (sympy-like) Functions.

        if not hasattr(m, "args") or not m.args:
            return m
        if level > max_level:
            return m

        head = m.__class__

        rev = []
        dont_go_deeper = True

        for x in m.args:
            if hasattr(x, "args") and x.args:
                dont_go_deeper = False
                break
            rev.append(x)

        if dont_go_deeper:
            if level in levels:
                return head(*reversed(rev))
            return head(*rev)

        if level in levels:
            return head(
                *(
                    cls.reverse(x, level + 1, levels, max_level)
                    for x in reversed(m.args)
                )
            )
        return head(*(cls.reverse(x, level + 1, levels, max_level) for x in m.args))

    @classmethod
    def exec(cls, x, levels=List(s.S.One)):
        if not isinstance(levels, iterables):
            levels = List(levels)

        if not all(x.is_number and x.is_integer and x > 0 for x in levels):
            raise FunctionException("Reverse::level")

        return cls.reverse(x, 1, levels, Max(levels))
