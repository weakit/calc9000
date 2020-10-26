from collections import OrderedDict
from math import ceil

import sympy as s

from calc9000 import references as r
from calc9000.custom import (List, ListException, NoOutput, Rule,
                             RuleException, Span, SpecialOutput, String, Tag)

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
                    raise FunctionException(
                        "General::thread", "Cannot Thread over Lists of Unequal Length."
                    )
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

    return type(name, (NormalFunction,), {"exec": fun})


def r_threaded(name, func):
    def fun(x):
        return r_thread(func, x)

    return type(name, (NormalFunction,), {"exec": fun})


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
    if hasattr(n, "is_Integer"):
        return bool(n.is_integer)
    if hasattr(n, "is_integer"):
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
        raise FunctionException("General::options", f"Unexpected option {arg.lhs}")
    return True


def options(args, ops: dict, defaults=None):
    ret = {}
    for arg in args:
        if not isinstance(arg.lhs, s.Symbol) or arg.lhs.name not in ops:
            raise FunctionException("General::options", f"Unexpected option {arg.lhs}")
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
        if hasattr(refs.BuiltIns[symbol], "tags") and tag in refs.BuiltIns[symbol].tags:
            return refs.BuiltIns[symbol].tags[tag]
    t = Tag(f"{symbol}::{tag}")
    if t in refs.TagValues:
        return refs.TagValues[t]
    return t


def message(tag, e):
    r.refs.add_message(tag, e)


def exec_func(cls, *args, **kwargs):
    clear_cache = False
    if hasattr(cls, "exec"):
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
                if cls.param_spec and not (
                    cls.param_spec[0] <= len(args_to_pass) <= cls.param_spec[1]
                ):
                    if cls.param_spec[0] == cls.param_spec[1]:
                        st = (
                            f"{cls.__name__} takes {cls.param_spec[0]} "
                            f"arguments but was given {len(args_to_pass)}."
                        )
                    else:
                        st = (
                            f"{cls.__name__} takes {cls.param_spec[0]} to {cls.param_spec[1]} "
                            f"arguments but was given {len(args_to_pass)}."
                        )
                    raise TypeError(st)

                # pass args and options as kws
                kws.update(kwargs)  # for internal calls
                return cls.exec(*args_to_pass, **kws)
            return cls.exec(*args, **kwargs)

        except FunctionException as x:
            message(x.tag, x.message)
            clear_cache = True
            return None

        except ValueError as v:
            message(f"{cls.__name__}::PythonValueError", str(v))
            clear_cache = True
            return None

        except TypeError as t:
            if str(t).startswith("exec()"):
                t = str(t).replace("exec()", cls.__name__)
                t = t.translate(
                    str.maketrans({x: str(int(x) - 1) for x in filter(str.isdigit, t)})
                )
                message(f"{cls.__name__}::PythonArgs", str(t))
            else:
                message(f"{cls.__name__}::PythonTypeError", str(t))
            clear_cache = True
            return None

        except NotImplementedError as e:
            message(f"{cls.__name__}::NotImplementedError", str(e))
            clear_cache = True
            return None

        except RecursionError as e:
            message(f"General::PythonRecursionError", str(e))
            clear_cache = True
            return None

        except ListException as e:
            message("List", e.args[0])
            clear_cache = True
            return None

        except RuleException as e:
            message("Rule", e.args[0])
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
            cond = (
                isinstance(other, s.Function)
                and str(self.__class__) == str(other.__class__)
                and all(
                    deepflatten(
                        thread(
                            lambda x, y: s.Eq(x, y) in (True, BooleanTrue),
                            self.args,
                            other.args,
                        )
                    )
                )
            )
            return cond or None
        except FunctionException:
            return None

    def __eq__(self, other):
        return (
            isinstance(other, s.Function)
            and str(self.__class__) == str(other.__class__)
            and self.args == other.args
        )

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


def evaluate_no_lazy(expr):
    """
    Evaluate an expression with current definitions.
    Does not evaluate any LazyFunctions in given expression.
    """

    # if is iterable, apply for each element
    if isinstance(expr, iterables) and not isinstance(expr, s.Matrix):
        return List.create(evaluate_no_lazy(x) for x in expr)

    # if expr does not have ability to perform substitutions, return
    if not hasattr(expr, "xreplace"):
        return expr

    def extend(ex):
        # if ex is a number, return
        if hasattr(ex, "is_Number") and ex.is_Number:
            return ex

        # if ex is a symbol, return definitions if present
        if isinstance(ex, s.Symbol):
            st = ex.name
            if st in r.refs.OwnValues:
                return r.refs.OwnValues[st]
            elif st in r.refs.Constants.Dict:
                return r.refs.Constants.Dict[st]

        # if ex does not have any arguments, return
        if not hasattr(ex, "args") or not isinstance(ex.args, iterables):
            return ex

        # ex is supposed to be a function at this point (since args are present)
        f_name = type(ex).__name__

        # do not make substitutions if function is explicit
        # since explicit functions require unevaluated args
        if Functions.is_explicit(f_name):
            return ex

        # evaluate args
        rep = {x: extend(x) for x in ex.args}
        ex = ex.xreplace(rep)

        # built-ins are automatically re-evaluated
        # when replacements are made by sympy

        # apply definitions if function has any
        if Functions.is_not_builtin(f_name):
            ex = Functions.apply_definitions(ex)

        return ex

    # evaluate inside-out
    return extend(expr)


class LazyFunction(s.Function):
    """
    A really bad Lazy Function implementation
    """

    @staticmethod
    def evaluate(expr):
        """
        Evaluate an expression with current definitions.
        Evaluates all LazyFunctions in given expression.
        """

        # see evaluate_no_lazy for explanation

        if isinstance(expr, iterables) and not isinstance(expr, s.Matrix):
            return List.create(LazyFunction.evaluate(x) for x in expr)

        if not hasattr(expr, "subs"):
            return expr

        def extend(ex):
            if hasattr(ex, "is_Number") and ex.is_Number:
                return ex

            if isinstance(ex, s.Symbol):
                st = ex.name
                if st in r.refs.OwnValues:
                    return r.refs.OwnValues[st]
                elif st in r.refs.Constants.Dict:
                    return r.refs.Constants.Dict[st]

            if not hasattr(ex, "args") or not isinstance(ex.args, iterables):
                return ex

            f_name = type(ex).__name__

            if Functions.is_explicit(f_name):
                # if explicit lazy function, call with args un-evaluated
                if isinstance(ex, LazyFunction):
                    return ex.land()
                return ex

            rep = {x: extend(x) for x in ex.args}
            ex = ex.xreplace(rep)

            if Functions.is_not_builtin(f_name):
                ex = Functions.apply_definitions(ex)

            # if lazy function, call actual function
            if isinstance(ex, LazyFunction):
                return ex.land()

            return ex

        return extend(expr)

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

    @classmethod
    def exec(cls):
        return None


class ArgsPatternSymbolPlaceholder:
    def __hash__(self):
        return self.type.__hash__() + 1

    def __init__(self, pat):
        # self.pat: raw pattern (as str)
        # self.subs_var: name of variable to be substituted
        # self.type: type of variable required

        self.pat = pat
        pat = pat.split("_")

        if len(pat) > 2:
            raise NotImplementedError(
                f"Patterns of type {self.pat} are not (yet) supported."
            )

        if pat[0]:
            self.subs_var = s.Symbol(pat[0])
        else:
            self.subs_var = None
        self.type = pat[1] or None

    def matches(self, var):
        if self.type is None:
            return True
        return Head.get_head(var) == self.type

    def __repr__(self):
        return self.pat


def rdp_check_eq(a, b):
    """helper function for remove_duplicate_pattern()"""
    if isinstance(a, ArgsPatternSymbolPlaceholder) and isinstance(
        b, ArgsPatternSymbolPlaceholder
    ):
        return a.type == b.type
    return a == b


def remove_duplicate_pattern(d, check_pat):
    """Removes duplicate ArgPatterns if present """
    for pat in d.keys():
        if hash(pat) == hash(check_pat) and all(
            rdp_check_eq(x, y) for x, y in zip(pat.prototype, check_pat.prototype)
        ):
            del d[pat]

            # only one duplicate should exist since
            # duplicates are removed at every assignment
            return


class ArgsPattern:
    def __hash__(self):
        return self.prototype.__hash__() + 1

    def __init__(self, *args):
        args = (LazyFunction.evaluate(x) for x in args)

        self.explicit_args = 0  # number of explicit args
        self.type_args = 0  # number of args with specified types
        self.prototype = []
        self.replacements = {}  # positions in prototypes that require replacements
        self.is_pattern = False

        for arg in args:

            if isinstance(arg, s.Symbol):
                if "_" in arg.name:

                    self.is_pattern = True
                    placeholder = ArgsPatternSymbolPlaceholder(arg.name)

                    if placeholder.type:
                        self.type_args += 1

                    if placeholder.subs_var:
                        self.replacements[len(self.prototype)] = placeholder

                    self.prototype.append(placeholder)
                    continue

            self.explicit_args += 1
            self.prototype.append(arg)

        self.prototype_length = len(self.prototype)  # cached length
        self.prototype = tuple(self.prototype)  # for safety

    @property
    def importance(self):
        return self.explicit_args, self.type_args

    def __len__(self):
        return self.prototype.__len__()

    def __iter__(self):
        return self.prototype.__iter__()

    def match(self, pat):
        if len(pat) != self.prototype_length:
            return False  # return False if size does not match

        # evaluate to get current definitions
        current_prototype = evaluate_no_lazy(self.prototype)

        if self.is_pattern:
            for x, y in zip(current_prototype, pat):
                if x != y:
                    if isinstance(x, ArgsPatternSymbolPlaceholder) and x.matches(y):
                        continue
                    return False
            return True

        # if not pattern, compare directly
        return all(x == y for x, y in zip(current_prototype, pat))

    def subs_dict(self, pat):
        pat = list(pat)
        subs_dict = {}
        for pos, placeholder in self.replacements.items():
            subs_dict[placeholder.subs_var] = pat[pos]
        return subs_dict

    def __repr__(self):
        return f"ArgsPattern{self.prototype}"


class SemicolonStatement(ExplicitFunction):
    @classmethod
    def exec(cls, expr):
        return NoOutput(LazyFunction.evaluate(expr))


def set_tag_value(tag: Tag, m):
    """Handles tag assignment"""
    # TODO: Redo
    # TODO: Store custom tags in references

    if not isinstance(m, String):
        raise FunctionException("Set::set_tag", f"{tag} can ony be set to a string.")
    refs = r.refs
    if tag.symbol in refs.BuiltIns:
        if hasattr(refs.BuiltIns[tag.symbol], "tags"):
            refs.BuiltIns[tag.symbol].tags[tag.tag] = m.value
    else:
        refs.TagValues[tag] = m


def is_assignable(f) -> bool:
    """returns True if a symbol/function can be assigned to"""
    f = str(f)
    return (
        f not in r.refs.BuiltIns
        and f not in r.refs.Constants.Dict
        and f not in r.refs.Protected.Dict
    )


def set_part_low(x, part, n):
    """returns var to be assigned"""

    head = x.__class__
    args = list(x.args)
    part = LazyFunction.evaluate(part)
    err_part = part  # for displaying errors

    if part == s.S.Zero:
        return Functions.call(n.name, *args)

    if isinstance(part, Span):
        part = part.slice()

        part_length = ceil((part.stop - (part.start or 0)) / (part.step or 1))

        # if not iterable or length does not match, assign all indices the same value
        if not isinstance(n, iterables) or part_length != len(n):
            n = (*(n for _ in range(part_length)),)

        # TODO: make sure args has part
    elif is_integer(part):
        part = int(part) - 1 if part > 0 else int(part)
    else:
        raise FunctionException(
            "Set::psetspec", f"{part} is not a valid Part specification."
        )

    try:
        args[part] = n
    except IndexError:
        raise FunctionException("Set::psetindex", f"{x} does not have part {err_part}")

    return head(*args)


def set_part(x, n):
    """Handles part assignment"""

    OwnValues = r.refs.OwnValues

    part = x.args[1]
    var = x.args[0]

    # make sure var is a symbol
    if not isinstance(var, s.Symbol):
        raise FunctionException(
            "Set::psets", f"{var} in part assignment is not a symbol."
        )

    # check if symbol var can be assigned to
    if not is_assignable(var.name):
        raise FunctionException(
            "Set::setx", f"Symbol {var} is protected and cannot be assigned to."
        )

    # make sure var has a value
    if var.name not in OwnValues:
        raise FunctionException(
            "Set::psetx",
            f"{var.name} does not have a value to be used in Part assignment.",
        )

    # TODO: Finish
    if len(x.args) > 2:
        raise NotImplementedError("Successive part assignment is not yet supported.")

    # part assignment for single part

    OwnValues[var.name] = set_part_low(OwnValues[var.name], part, n)
    return n


def do_set(x, n):
    """Handles assignment"""
    refs = r.refs

    if isinstance(x, s.Symbol):
        if not is_assignable(x.name):
            raise FunctionException(
                "Set::setx", f"Symbol {x} is protected and cannot be assigned to."
            )

        if isinstance(x, Tag):
            set_tag_value(x, n)
            return n

        if isinstance(n, s.Expr):
            if x in n.atoms():
                return None  # TODO: raise

        refs.OwnValues[x.name] = n

        return n

    if isinstance(x, s.Function):
        name = type(x).__name__

        # handle part assignment
        if name == "Part":
            return set_part(x, n)

        if not is_assignable(name):
            raise FunctionException("Set::set", f"Symbol {x} cannot be Assigned to.")

        # create dict of patterns if not present
        if name not in refs.DownValues:
            refs.DownValues[name] = {ArgsPattern(*x.args): n}

        # else add pattern to existing dict
        else:
            pat = ArgsPattern(*x.args)

            # remove duplicate if present
            remove_duplicate_pattern(refs.DownValues[name], pat)

            # update in place
            refs.DownValues[name].update({pat: n})

            # sort dict when done
            # sorting is done during assignment to keep function calls fast
            refs.DownValues[name] = OrderedDict(
                sorted(
                    refs.DownValues[name].items(),
                    key=lambda z: z[0].importance,
                    reverse=True,
                )
            )

        return n

    if isinstance(x, iterables):
        if isinstance(n, iterables) and len(x) == len(n):
            return List.create(Set(a, b) for a, b in zip(x, n))
        else:
            raise FunctionException("Set::shape")

    return None


class Set(ExplicitFunction):
    """
    Set [x, n]
     x = n
     Sets a symbol x to have the value n.
    """

    @classmethod
    def exec(cls, x, n):
        return do_set(x, LazyFunction.evaluate(n))


class Unset(ExplicitFunction):
    """
    Unset [x]
    x =.
     Deletes a symbol or list of symbols x,
     if they were previously assigned a value.
    """

    @classmethod
    def exec(cls, n):
        if isinstance(n, iterables):
            return List.create(Unset(x) for x in n)
        if isinstance(n, s.Symbol) and str(n) in r.refs.OwnValues:
            del r.refs.OwnValues[str(n)]
        return NoOutput(None)


class Clear(ExplicitFunction):
    """
    Clear [x, y, …]
     Deletes all definitions for x, y, ….
    """

    @staticmethod
    def do_clear(n):
        if isinstance(n, s.Symbol):
            name = n.name
            if name in r.refs.OwnValues:
                del r.refs.OwnValues[name]
            if name in r.refs.DownValues:
                del r.refs.DownValues[name]
            del name

    @classmethod
    def exec(cls, *args):
        for arg in args:
            cls.do_clear(arg)
        return NoOutput(None)


class SetDelayed(ExplicitFunction):
    @classmethod
    def exec(cls, x, n):
        value = do_set(x, Delay(n))

        if value:
            return value.args[0]

        return None


class Plus(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Add, *args)


class Times(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Mul, *args)


class Power(NormalFunction):
    @classmethod
    def exec(cls, *args):
        return thread(s.Pow, *args)


class Nothing(NormalFunction):
    @classmethod
    def exec(cls, *args, **kwargs):
        return r.refs.Constants.Nothing


class Head(NormalFunction):
    """
    Head [expr]
     Gives the head of expr.
    """

    @staticmethod
    def get_head(x):
        if x in (s.S.NegativeOne, s.S.Zero, s.S.One):
            return "Integer"
        if x == s.S.Half:
            return "Rational"
        return type(x).__name__

    @classmethod
    def exec(cls, h, f=None):
        if f is not None:
            return Functions.call(str(f), cls.get_head(h))
        return s.Symbol(cls.get_head(h))


class CompoundExpression(ExplicitFunction):
    """
    !internal
    """

    @classmethod
    def exec(cls, *args):
        for expr in args[:-1]:
            LazyFunction.evaluate(expr)
        return LazyFunction.evaluate(args[-1])


class Replace(NormalFunction):
    """
    Replace [Expr, Rule]
     Transforms Expression expr with the given Rule.

    Replace [Expr, {Rule1, Rule2, …}]
     Transforms Expression expr with the given Rules.
    """

    @staticmethod
    def func_replacement_helper(replacements):
        if isinstance(replacements, dict):
            return replacements

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

        replacement_dict = Replace.func_replacement_helper(replacements)

        # TODO: (low) rewrite

        for func in expr.atoms(s.Function):
            if str(func) in replacement_dict:
                expr = expr.replace(func, replacement_dict[str(func)])
            if str(func.func) in replacement_dict:
                expr = expr.replace(
                    func,
                    Functions.call(str(replacement_dict[str(func.func)]), *func.args),
                )

        # for lists
        expr = expr.replace(s.Add, Plus)
        expr = expr.replace(s.Mul, Times)

        return LazyFunction.evaluate(expr)

    @classmethod
    def exec(cls, expr, replacements):
        if not isinstance(replacements, iterables):
            replacements = (replacements,)
        else:
            if isinstance(replacements[0], iterables):
                if not all(isinstance(x, iterables) for x in replacements):
                    raise FunctionException(
                        "Replace::subs",
                        f"{replacements} is a mixture of Lists and Non-Lists.",
                    )
                return List(
                    *(cls.exec(expr, replacement) for replacement in replacements)
                )
            else:
                if not all(not isinstance(x, iterables) for x in replacements):
                    raise FunctionException(
                        "Replace::subs",
                        f"{replacements} is a mixture of Lists and Non-Lists.",
                    )

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
        if isinstance(out, s.Expr):
            out = evaluate_no_lazy(out)
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

    # TODO: Norm
    # TODO: Prime Notation
    # TODO: Derivative
    # TODO: Part Assignment tests

    # TODO: Polar Complex Number Representation
    # TODO: Series
    # TODO: NSolve, DSolve
    # TODO: Roots (Solve)

    # TODO: Matrix Row Operations
    # TODO: Remaining Matrix Operations

    # TODO: Fix Compound Expressions (f[x][y][z])

    # TODO: Latex Printer
    # TODO: Relational Functions
    # TODO: References Storage

    # Low Priority

    # TODO: Try Piecewise
    # TODO: Map, Apply
    # TODO: Arithmetic Functions: Ratios, Differences
    # TODO: Booleans, Conditions, Boole
    # TODO: Cross of > 3 dimensional vectors
    # TODO: Implement Fully: Total, Quotient, Factor

    # for now, until I find something better
    r.refs.BuiltIns.update(
        {
            k: v
            for k, v in globals().items()
            if isinstance(v, type)
            and issubclass(v, s.Function)
            and not issubclass(v, LazyFunction)
        }
    )

    Customs = {"List": List, "Rule": Rule}

    @classmethod
    def not_normal(cls, f: str) -> bool:
        if f in r.refs.BuiltIns:
            return not issubclass(r.refs.BuiltIns[f], NormalFunction)
        return False

    @classmethod
    def is_not_builtin(cls, f: str) -> bool:
        return f not in r.refs.BuiltIns

    @classmethod
    def is_builtin(cls, f: str) -> bool:
        return f in r.refs.BuiltIns

    @classmethod
    def is_explicit(cls, f: str) -> bool:
        if f in r.refs.BuiltIns:
            return issubclass(r.refs.BuiltIns[f], ExplicitFunction)
        return False

    @classmethod
    def apply_definitions(cls, f):
        return cls._apply_definitions(r.refs, type(f).__name__, f.args) or f

    @staticmethod
    def _apply_definitions(refs, f, a):
        if f in refs.DownValues:
            # check all patterns
            patterns = refs.DownValues[f]
            for pattern in patterns:
                if pattern.match(a):
                    # get substitutions to make
                    subs_to_do = pattern.subs_dict(a)

                    # thread make subs (thread in-case is list)
                    ret = thread(Replace.do_subs, patterns[pattern], subs_to_do)

                    # evaluate before returning
                    # using string comparison because type comparison
                    # isn't working for some reason.
                    # TODO: Investigate
                    if str(type(ret)) == "Delay":
                        return LazyFunction.evaluate(ret.args[0])

                    return LazyFunction.evaluate(ret)

    @classmethod
    def call(cls, f: str, *a):
        refs = r.refs
        # clear cache if necessary

        if refs.CacheClearQueued:
            s.core.cache.clear_cache()
            refs.CacheClearQueued = False

        elif f in refs.NoCache or cls.not_normal(f):
            s.core.cache.clear_cache()

        # Check in BuiltIns first
        if f in refs.BuiltIns:
            return refs.BuiltIns[f](*a)

        if f in cls.Customs:
            return cls.Customs[f](*a)

        # Check DownValues
        defs = cls._apply_definitions(refs, f, a)

        if defs:
            return defs

        return type(f, (DefinedFunction,), {})(*a)

    @classmethod
    def lazy_call(cls, f: str, *a):
        return type(f, (LazyFunction,), {})(*a)
