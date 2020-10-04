from calc9000.functions.core import *
from iteration_utilities import deepflatten, accumulate
from itertools import permutations, combinations


class Part(NormalFunction):
    # TODO: Raise exception on {a, b, c}[[;;;;-1]]
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
                raise FunctionException('Part::dim', f'Part {n} of {expr} does not exist.')
        raise FunctionException('Part::part', f'{n} is not a valid Part specification.')

    @classmethod
    def exec(cls, expr, *args):
        part = head = None
        if not args:
            return expr
        if hasattr(expr, 'args'):
            part = expr.args
            head = expr.__class__
        elif hasattr(expr, '__getitem__'):
            part = expr
            head = List
        arg = args[0]
        if arg == s.S.Zero:
            return s.Symbol(type(expr).__name__)
        if not part:
            raise FunctionException('Part::dim', f'{expr} does not have Part {arg}')
        if arg == r.refs.Constants.All:  # TODO: add None
            arg = Range(len(expr))
        if isinstance(arg, Span):
            return Part(Take(expr, arg), *args[1:])  # pass expr with head
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
            if (0 in (lower, upper, step) or not (is_integer(lower) and is_integer(upper) and is_integer(step))) \
                    or len(seq) > 3:
                raise FunctionException('Take::dim', 'Invalid Bounds for Take.')
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
        raise FunctionException('Take::take', f'{seq} is not a valid Take specification.')

    @classmethod
    def exec(cls, expr, *seqs):
        take = head = None

        if not seqs:
            return expr
        if hasattr(expr, 'args'):
            take = expr.args
            head = expr.__class__
        elif hasattr(expr, '__getitem__'):
            take = expr
            head = List

        if len(seqs) > 1:
            return head(*(cls.exec(x, *seqs[1:]) for x in cls.get_take(take, seqs[0])))
        return head(*cls.get_take(take, seqs[0]))


class Head(NormalFunction):
    """
    Head [expr]
     Gives the head of expr.
    """

    @classmethod
    def exec(cls, h, f=None):
        if f is not None:
            return Functions.call(str(f), Part(h, 0))
        return Part(h, 0)


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
        ret = []
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
        return List(*ret)

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
        li = []
        for arg in args:
            li.append(LazyFunction.evaluate(Subs(expr, Rule(repl, arg))))
        return List.create(li)

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

        li = []
        for expr_, specs in zip(
                cls._range_parse(expr, args[0]),
                cls._range_parse(args[1:], args[0])
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
    """

    @classmethod
    def exec(cls, li, n_spec=None):
        subsets = []

        if n_spec is None:
            n_spec = range(len(li) + 1)
        elif n_spec.is_number:
            if not is_integer(n_spec):
                raise FunctionException('Subsets::exec', f'{n_spec} is not an integer.')
            n_spec = range(int(n_spec) + 1)
        else:
            n_spec = Range(*n_spec)

        for spec in n_spec:
            subsets += [List(*x) for x in combinations(li, spec)]

        return List(*subsets)


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


class Reverse(NormalFunction):
    """

    """

    tags = {
        'level': 'A positive integer is expected as a level.'
    }

    @classmethod
    def reverse(cls, m, level, levels, max_level):
        # only works with Lists and (sympy-like) Functions.

        if not hasattr(m, 'args') or not m.args:
            return m
        if level > max_level:
            return m

        head = m.__class__

        rev = []
        dont_go_deeper = True

        for x in m.args:
            if hasattr(x, 'args') and x.args:
                dont_go_deeper = False
                break
            rev.append(x)

        if dont_go_deeper:
            return head(*rev)

        if level in levels:
            return head(*(cls.reverse(x, level + 1, levels, max_level) for x in reversed(m.args)))
        return head(*(cls.reverse(x, level + 1, levels, max_level) for x in m.args))

    @classmethod
    def exec(cls, x, levels=List(s.S.One)):
        if not isinstance(levels, iterables):
            levels = List(levels)

        if not all(x.is_number and x.is_integer and x > 0 for x in levels):
            raise FunctionException('Reverse::level')

        return cls.reverse(x, 1, levels, Max(levels))

