import sympy as s
from calc9000 import references as r
from calc9000.custom import List, Rule, Tag, String, Span
from calc9000.custom import ListException, RuleException


iterables = (s.Tuple, List, s.Matrix, list, tuple)

ExtraPrecision = r.refs.ExtraPrecision
DefaultPrecision = r.refs.DefaultPrecision
WorkingPrecision = r.refs.WorkingPrecision


def thread(func, *args, **kwargs):
    """
    Internal threading function
    keyword args are not threaded
    """

    length = None

    for arg in args:
        if isinstance(arg, iterables):
            if length is not None:
                if length != len(arg):
                    raise FunctionException('General::thread', "Cannot Thread over Lists of Unequal Length.")
            else:
                length = len(arg)

    if length is None:
        return func(*args, **kwargs)

    chained = list(args)

    for i in range(len(chained)):
        if not isinstance(chained[i], iterables):
            chained[i] = (chained[i],) * length

    return List.create(thread(func, *z, **kwargs) for z in zip(*chained))

    # if isinstance(x, iterables):
    #     temp_list = List()
    #     for item in x:
    #         temp_list.append(thread(func, item))
    #     return temp_list
    # return func(x)


def r_thread(func, to_thread, *args, rule=True, **kwargs):
    if isinstance(to_thread, iterables):
        return List(*(r_thread(func, x, *args, **kwargs) for x in to_thread))

    if rule and isinstance(to_thread, Rule):
        return Rule(
            r_thread(func, to_thread.lhs, *args, **kwargs),
            r_thread(func, to_thread.rhs, *args, **kwargs),
        )

    return func(to_thread, *args, **kwargs)


def threaded(name, func):
    def fun(x):
        return thread(func, x)

    return type(name, (NormalFunction,), {'exec': fun})


def r_threaded(name, func):
    def fun(x):
        return r_thread(func, x)

    return type(name, (NormalFunction,), {'exec': fun})


def boolean(x):
    if isinstance(x, bool):
        return x
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


def is_integer(n):
    """Returns true if is integer."""
    if hasattr(n, 'is_Integer'):
        return bool(n.is_integer)
    if hasattr(n, 'is_integer'):
        return bool(n.is_integer)
    if isinstance(n, int):
        return True
    if isinstance(n, (s.Float, float)):
        return int(n) == n
    return False


class FunctionException(Exception):
    """General class for function related exceptions"""
    def __init__(self, tag: str, m=None):
        if not isinstance(tag, Tag):
            tag = Tag(tag)
        self.tag = tag
        self.message = m or get_tag_value(tag.symbol, tag.tag)


def in_options(arg, ops):
    if not isinstance(arg.lhs, s.Symbol) or arg.lhs.name not in ops:
        raise FunctionException('General::options', f"Unexpected option {arg.lhs}")
    return True


def options(args, ops: dict, defaults=None):
    ret = {}
    for arg in args:
        if not isinstance(arg.lhs, s.Symbol) or arg.lhs.name not in ops:
            raise FunctionException('General::options', f"Unexpected option {arg.lhs}")
        if str(arg.rhs) in ("True", "False"):
            arg.rhs = boolean(arg.rhs)
        ret[ops[arg.lhs.name]] = arg.rhs
    if defaults is not None:
        for default in defaults:
            if default not in ret:
                ret[default] = defaults[default]
    return ret


def get_symbol_value(n):
    """
    Returns value of symbol if present, else returns symbol.
    """
    refs = r.refs
    if n in refs.Constants.Dict:
        return refs.Constants.Dict[n]
    if n not in refs.OwnValues:
        return s.Symbol(n)
    ret = refs.OwnValues[n]
    if type(ret) is Delay:
        return LazyFunction.evaluate(ret.args[0])
    # TODO: Improve performance for large (especially float) lists
    return LazyFunction.evaluate(ret)


def get_tag_value(symbol, tag):
    symbol, tag = str(symbol), str(tag)
    refs = r.refs
    if symbol in refs.BuiltIns:
        if hasattr(refs.BuiltIns[symbol], 'tags') and tag in refs.BuiltIns[symbol].tags:
            return refs.BuiltIns[symbol].tags[tag]
    t = Tag(f'{symbol}::{tag}')
    if t in refs.TagValues:
        return refs.TagValues[t]
    return t


def message(tag, e):
    r.refs.add_message(tag, e)


def exec_func(cls, *args, **kwargs):
    clear_cache = False
    if hasattr(cls, 'exec'):
        try:
            if cls.op_spec:  # check if function accepts options

                # find first rule (option) occurrence
                i = len(args)
                for i, v in enumerate(reversed(args)):
                    if not isinstance(v, Rule):
                        i -= 1
                        break
                i = max(cls.param_spec[0], len(args) - i - 1)

                # take remaining arguments and separate options
                args_to_pass = args[:i]
                kws = options(args[i:], *cls.op_spec)

                # check params and raise error if no of args is invalid
                if cls.param_spec and not (cls.param_spec[0] <= len(args_to_pass) <= cls.param_spec[1]):
                    if cls.param_spec[0] == cls.param_spec[1]:
                        st = f'{cls.__name__} takes {cls.param_spec[0]} ' \
                             f'arguments but was given {len(args_to_pass)}.'
                    else:
                        st = f'{cls.__name__} takes {cls.param_spec[0]} to {cls.param_spec[1]} ' \
                             f'arguments but was given {len(args_to_pass)}.'
                    raise TypeError(st)

                # pass args and options as kws
                kws.update(kwargs)  # for internal calls
                return cls.exec(*args_to_pass, **kws)
            return cls.exec(*args, **kwargs)

        except FunctionException as x:
            message(x.tag,  x.message)
            clear_cache = True
            return None

        except ValueError as v:
            message(f'{cls.__name__}::PythonValueError', str(v))
            clear_cache = True
            return None

        except TypeError as t:
            if str(t).startswith('exec()'):
                t = str(t).replace('exec()', cls.__name__)
                t = t.translate(str.maketrans({x: str(int(x) - 1) for x in filter(str.isdigit, t)}))
                message(f'{cls.__name__}::PythonArgs', str(t))
            else:
                message(f'{cls.__name__}::PythonTypeError', str(t))
            clear_cache = True
            return None

        except NotImplementedError as e:
            message(f'{cls.__name__}::NotImplementedError', str(e))
            clear_cache = True
            return None

        except ListException as e:
            message('List', e.args[0])
            clear_cache = True
            return None

        except RuleException as e:
            message('Rule', e.args[0])
            clear_cache = True
            return None

        finally:
            if clear_cache:
                r.refs.CacheClearQueued = True
    return None


class NormalFunction(s.Function):
    """
    Ordinary Function Class
    Works for most Functions.
    """

    op_spec = None
    param_spec = (0, s.oo)
    rule_param = False

    @classmethod
    def eval(cls, *args, **kwargs):
        return exec_func(cls, *args, **kwargs)


class DefinedFunction(NormalFunction):
    def _eval_Eq(self, other):
        from iteration_utilities import deepflatten
        from sympy.logic.boolalg import BooleanTrue
        # TODO: figure out something better
        try:
            check = deepflatten(thread(
                    lambda x, y: s.Eq(x, y) in (True, BooleanTrue),
                    self.args, other.args))
            cond = isinstance(other, s.Function) and str(self.__class__) == str(other.__class__) and all(check)
            return cond or None
        except FunctionException:
            return None

    def __eq__(self, other):
        return bool(self._eval_Eq(other))

    def __hash__(self):
        return hash((self.class_key(), frozenset(self.args)))


class ExplicitFunction(s.Function):
    """
    Functions that need to be called with the arguments unevaluated.
    """
    op_spec = None
    param_spec = (0, s.oo)
    rule_param = False

    @classmethod
    def eval(cls, *args):
        return exec_func(cls, *args)


class LazyFunction(s.Function):
    """
    A really bad Lazy Function implementation
    """

    @staticmethod
    def evaluate(expr):
        """Get value of Lazy Function, with other lazy functions as args."""
        if isinstance(expr, iterables) and not isinstance(expr, s.Matrix):
            return List.create(LazyFunction.evaluate(x) for x in expr)

        if not hasattr(expr, 'subs'):
            return expr

        def extend(ex):
            # TODO: replace DownValues, etc.

            if hasattr(ex, 'is_Number') and ex.is_Number:
                return ex

            if isinstance(ex, s.Symbol):
                st = str(ex)
                if st in r.refs.OwnValues:
                    return r.refs.OwnValues[st]
                elif st in r.refs.Constants.Dict:
                    return r.refs.Constants.Dict[st]

            if not hasattr(ex, 'args') or not isinstance(ex.args, iterables):
                return ex

            if Functions.is_explicit(type(ex).__name__):
                if isinstance(ex, LazyFunction):
                    return ex.land()
                return ex

            rep = {x: extend(x) for x in ex.args}
            ex = ex.xreplace(rep)

            if isinstance(ex, LazyFunction):
                return ex.land()

            return ex

        expr = extend(expr)

        return expr

    def land(self):
        return Functions.call(type(self).__name__, *self.args)

    @classmethod
    def exec(cls):
        return None


class Delay(LazyFunction):
    """
    A delayed expression.
    Use first arg as expression.
    """


class SemicolonStatement(ExplicitFunction):
    """
    !internal
    """

    @classmethod
    def exec(cls, expr):
        return r.NoOutput(LazyFunction.evaluate(expr))


def set_tag_value(tag: Tag, m):
    if not isinstance(m, String):
        raise FunctionException('Set::set_tag', f'{tag} can ony be set to a string.')
    refs = r.refs
    if tag.symbol in refs.BuiltIns:
        if hasattr(refs.BuiltIns[tag.symbol], 'tags'):
            refs.BuiltIns[tag.symbol].tags[tag.tag] = m.value
    else:
        refs.TagValues[tag] = m


def real_set(x, n):
    refs = r.refs
    for ref in [
        refs.Constants.Dict,
        refs.BuiltIns,
        refs.Protected.Dict
    ]:
        if str(x) in ref:
            raise FunctionException('Set::set', f'Symbol {x} cannot be Assigned to.')
    if isinstance(x, s.Symbol):
        if isinstance(x, Tag):
            set_tag_value(x, n)
            return n
        if isinstance(n, s.Expr):
            if x in n.atoms():
                return None
        refs.OwnValues[x.name] = n
        return n
    if isinstance(x, s.Function):
        # process pattern
        f_args = []
        name = type(x).__name__
        expr = n
        num = 1
        for arg in x.args:
            if isinstance(arg, s.Symbol) and arg.name.endswith('_'):
                expr = Subs(expr, Rule(s.Symbol(arg.name[:-1]), s.Symbol(f'*{num}')))
                f_args.append(s.Symbol(f'*{num}'))
                num += 1
            else:
                f_args.append(arg)
        # create patterns list if not present
        if name not in refs.DownValues:
            refs.DownValues[name] = {tuple(f_args): (expr, ())}
        else:  # else add pattern to list
            refs.DownValues[name].update({tuple(f_args): (expr, ())})
        return n
    if isinstance(x, iterables):
        if isinstance(x, iterables) and len(x) == len(n):
            return List.create(Set(a, b) for a, b in zip(x, n))
    return None


class Set(ExplicitFunction):
    """
    Set [x, n]
     x = n
     Sets a symbol x to have the value n.
    """

    @classmethod
    def exec(cls, x, n):
        return real_set(x, LazyFunction.evaluate(n))


class Unset(NormalFunction):
    """
    Unset [x]
    x =.
        Deletes a symbol or list of symbols x, if they were previously assigned a value.
    """

    @classmethod
    def exec(cls, n):
        if isinstance(n, iterables):
            return List.create(Unset(x) for x in n)
        if isinstance(n, s.Symbol) and str(n) in r.refs.OwnValues:
            del r.refs.OwnValues[str(n)]
        # TODO: return 'Nothing' when done
        return None


class SetDelayed(ExplicitFunction):
    @classmethod
    # TODO: Delayed function set?
    def exec(cls, x, n):
        value = real_set(x, Delay(n))
        if hasattr(value, 'args'):
            return value.args[0]
        return None


# def DelayedSet(f, x, n):
#     # TODO: again
#     refs = r.refs
#     for ref in [
#         refs.Constants.Dict,
#         refs.BuiltIns,
#         refs.Protected.Dict
#     ]:
#         if str(x) in ref:
#             raise FunctionException(f'Symbol {x} cannot be Assigned to.')
#     if isinstance(x, s.Symbol):
#         if isinstance(n, s.Expr):
#             if x in n.atoms():
#                 return None
#         refs.Symbols[x.name] = n
#         return n
#     if isinstance(x, s.Function):
#         list_ = []
#         name = type(x).__name__
#         expr = n
#         num = 1
#         for arg in x.args:
#             if isinstance(arg, s.Symbol) and arg.name.endswith('_'):
#                 expr = Subs(expr, Rule(s.Symbol(arg.name[:-1]), s.Symbol(f'*{num}')))
#                 list_.append(s.Symbol(f'*{num}'))
#                 num += 1
#             else:
#                 list_.append(arg)
#         if name not in refs.Functions:
#             refs.Functions[name] = {tuple(list_): (expr, f)}
#         else:
#             refs.Functions[name].update({tuple(list_): (expr, f)})
#         return n
#     if isinstance(x, iterables):
#         if isinstance(x, iterables) and len(x) == len(n):
#             return List.create(DelayedSet(f, a, b) for a, b in zip(x, n))


class Nothing(NormalFunction):
    @classmethod
    def exec(cls, *args, **kwargs):
        return r.refs.Constants.Nothing


class CompoundExpression(ExplicitFunction):
    """
    !internal
    """

    @classmethod
    def exec(cls, *args):
        for expr in args[:-1]:
            LazyFunction.evaluate(expr)
        return LazyFunction.evaluate(args[-1])


class Subs(NormalFunction):
    """
    Subs [Expr, Rule]
     Transforms Expression expr with the given Rule.

    Subs [Expr, {Rule1, Rule2, …}]
     Transforms Expression expr with the given Rules.
    """

    @staticmethod
    def func_replacement_helper(replacements):
        reps = {str(k): v for k, v in replacements}
        fw = r.refs.FunctionWrappers
        for x in list(reps):
            if x in fw:
                reps[fw[x]] = reps[x]
            elif x in fw.values():
                del reps[x]
        return reps

    @classmethod
    def do_subs(cls, expr, replacements):
        expr = expr.subs(replacements)

        replacement_dict = Subs.func_replacement_helper(replacements)

        # TODO: (low) rewrite

        for func in expr.atoms(s.Function):
            if str(func) in replacement_dict:
                expr = expr.replace(func, replacement_dict[str(func)])
            if str(func.func) in replacement_dict:
                expr = expr.replace(func, Functions.call(str(replacement_dict[str(func.func)]), *func.args))

        return LazyFunction.evaluate(expr)

    @classmethod
    def exec(cls, expr, replacements):
        if not isinstance(replacements, iterables):
            replacements = (replacements,)
        else:
            if isinstance(replacements[0], iterables):
                if not all(isinstance(x, iterables) for x in replacements):
                    raise FunctionException('Subs::subs', f'{replacements} is a mixture of Lists and Non-Lists.')
                return List(*(cls.exec(expr, replacement) for replacement in replacements))
            else:
                if not all(not isinstance(x, iterables) for x in replacements):
                    raise FunctionException('Subs::subs', f'{replacements} is a mixture of Lists and Non-Lists.')

        if isinstance(expr, iterables) and not isinstance(expr, s.Matrix):
            return List(*(cls.do_subs(x, replacements) for x in expr))

        return cls.do_subs(expr, replacements)


class N(NormalFunction):
    """
    N [expr]
     Gives the numerical value of expr.

    N [expr, n]
     Attempts to give a result with n-digit precision.

    Equivalent to sympy.N().
    """

    @classmethod
    def exec(cls, expr, p=DefaultPrecision, *args):
        # return thread(lambda x: s.N(x, *args), n)
        return r_thread(s.N, expr, p + ExtraPrecision, *args)


class In(NormalFunction):
    """
    In [n]
     Gives the raw input given in the nth line.
    """

    @staticmethod
    def _in(n):
        if n is None:
            return String(r.refs.get_in())
        if is_integer(n):
            if 0 < n < r.refs.Line:
                return String(r.refs.get_in(n))
            if -r.refs.Line < n < 0:
                return String(r.refs.get_in(r.refs.Line + n))
        return None

    @classmethod
    def exec(cls, n=None):
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
        if is_integer(n):
            if 0 < n < r.refs.Line:
                out = r.refs.get_out(n)
            elif -r.refs.Line < n < 0:
                out = r.refs.get_out(r.refs.Line + n)
        if isinstance(out, s.Expr):  # TODO: Replace with Subs func.
            out = out.subs(r.refs.OwnValues)
        return out

    @classmethod
    def exec(cls, n=None):
        return thread(cls.out, n)


# TODO: Proper Logic Functions w/ True and False


class And(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.And(*args)


class Or(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Or(*args)


class Not(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Not(*args)


class Xor(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Xor(*args)


class Nor(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Nor(*args)


class Nand(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Nand(*args)


class Implies(NormalFunction):
    @classmethod
    def exec(cls, p, q):
        return s.Implies(p, q)


class Equivalent(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return s.Equivalent(*args)

# TODO: Relational Functions

# class Boole(NormalFunction):
#     @classmethod
#     def exec(cls, x):
#         if isinstance(x, iterables):
#             return List.create(Boole(i) for i in x)
#         if isinstance(x, (bool, BooleanTrue, BooleanFalse)):
#             if x:
#                 return 1
#             return 0
#         return None


class Functions:
    # TODO: Finish Explicit Functions
    # TODO: Convert NormalFunctions to ExplicitFunctions
    # TODO: Double Check
    # TODO: Figure out Custom Functions

    # TODO: Norm
    # TODO: Prime Notation
    # TODO: Part, Assignment + Part Assignment tests
    # TODO: Integral, Derivative

    # TODO: Polar Complex Number Representation
    # TODO: Series
    # TODO: NSolve, DSolve
    # TODO: Roots (Solve)
    # TODO: Unit Conversions

    # TODO: Simple List Functions

    # TODO: Make Matrix Functions use List
    # TODO: Matrix Representation
    # TODO: Matrix Row Operations
    # TODO: Remaining Matrix Operations

    # TODO: Fix Compound Expressions (f[x][y][z])

    # TODO: Clear Function from References

    # TODO: Latex Printer
    # TODO: Float Precision
    # TODO: References Storage

    # Low Priority

    # TODO: Try Piecewise
    # TODO: Map, Apply
    # TODO: Arithmetic Functions: Ratios, Differences
    # TODO: Booleans, Conditions, Boole
    # TODO: Cross of > 3 dimensional vectors
    # TODO: Implement Fully: Total, Quotient, Factor

    # for now, until I find something better
    r.refs.BuiltIns.update({k: v for k, v in globals().items() if isinstance(v, type) and issubclass(v, s.Function)
                            and not issubclass(v, LazyFunction)})

    @classmethod
    def not_normal(cls, f: str) -> bool:
        if f in r.refs.BuiltIns:
            return not issubclass(r.refs.BuiltIns[f], NormalFunction)
        return False

    @classmethod
    def is_explicit(cls, f: str) -> bool:
        if f in r.refs.BuiltIns:
            return issubclass(r.refs.BuiltIns[f], ExplicitFunction)
        return False

    @classmethod
    def call(cls, f: str, *a):
        refs = r.refs
        # clear cache if necessary
        if refs.CacheClearQueued:
            s.core.cache.clear_cache()
            refs.CacheClearQueued = False
        elif f in refs.NoCache or cls.not_normal(f):
            s.core.cache.clear_cache()
        if f in refs.BuiltIns:
            return refs.BuiltIns[f](*a)
        # if f in refs.DownValues:
        #     priority = {}
        #     for header in list(refs.DownValues[f])[::-1]:
        #         match = True
        #         matches = 0
        #         if len(a) != len(header):
        #             continue
        #         for ar, br in zip(a, header):
        #             if ar == br:
        #                 matches += 1
        #                 continue
        #             if isinstance(br, s.Symbol) and br.name.startswith('*'):
        #                 continue
        #             match = False
        #             break
        #         if match:
        #             priority[matches] = header
        #     if priority:
        #         header = priority[max(priority)]
        #         expr = refs.DownValues[f][header][0]
        #         # reps = refs.Functions[f][header][1]
        #         for ar, br in zip(a, header):
        #             if isinstance(br, s.Symbol) and br.name.startswith('*'):
        #                 expr = Subs(expr, Rule(br, ar))
        #         expr = Subs(expr, Rule.from_dict(vars(r.refs.Symbols)))  # + Rule.from_dict({x: x for x in reps}))
        #         if type(expr) is Delay:
        #             # TODO: improve naive approach
        #             return LazyFunction.evaluate(expr.args[0])
        #         return expr
        return type(f, (DefinedFunction,), {})(*a)

    @classmethod
    def lazy_call(cls, f: str, *a):
        return type(f, (LazyFunction,), {})(*a)
