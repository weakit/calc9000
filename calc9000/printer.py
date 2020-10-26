import sympy as s
from mpmath.libmp.libmpf import dps_to_prec
from sympy.printing.pretty.pretty import (PRECEDENCE, PrettyPrinter,
                                          precedence_traditional, pretty_atom,
                                          prettyForm, sstr, stringPict)

from calc9000.functions.__init__ import Cross, DefinedFunction, Dot, Limit
from calc9000.references import FunctionWrappersReverse, refs


class ListSkip:
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"··· {self.n} skipped elements ···"

    def __repr__(self):
        return self.__str__()


class Printer9000(PrettyPrinter):
    _float_cutoff = dps_to_prec(refs.ExtraPrecision) - 4

    def _print_Float(self, e):
        # TODO: fix float printing in List
        e = s.Float(e, precision=max(e._prec - self._float_cutoff, 1))
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))

    def _print_Range(self, e):
        if isinstance(e, s.Range):
            return super()._print_Range(e)
        return super()._print_Function(e)

    def _print_Dot(self, e):
        if isinstance(e, Dot):
            return self._print_seq(
                e.args,
                None,
                None,
                ".",
                parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"],
            )
        return super()._print_Dot(e)

    def _print_Cross(self, e):
        if isinstance(e, Cross):
            return self._print_seq(
                e.args,
                None,
                None,
                "×",
                parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"],
            )
        return super()._print_Cross(e)

    # def _print_Rule(self, e):
    #     TODO: Proper rule printing
    #     return self._print_Implies(List(
    #         self._print_seq(e.lhs),
    #         self._print_seq(e.rhs)
    #     ), altchar='->')

    def _print_Limit(self, lim):
        if isinstance(lim, Limit):
            return super()._print_Function(lim)
        return super()._print_Limit(lim)

    @staticmethod
    def pretty_int(x):
        x = str(x)
        i = len(x) % 3

        for digit in x:
            if i > 0:
                yield digit
                i -= 1
            else:
                i = 2
                yield " "
                yield digit

    # def _print_Integer(self, x):
    #     pretty_int = self.pretty_int(x)
    #     final_str = ''.join(*pretty_int).strip()
    #     prettyForm(final_str)

    def _print_List(self, e):
        # for better performance
        if len(e) > 10:
            avg_len = (
                sum([len(pretty_print(e[x])) for x in range(0, len(e), len(e) // 10)])
                / 10
            )
            avg_total_len = (avg_len + 2) * len(e) + 2
            if avg_total_len > 1000:
                m = int(75 / avg_len)
                return self._print_seq(
                    e.value[:m] + [ListSkip(len(e.value) - 2 * m)] + e.value[-m:],
                    "{",
                    "}",
                )
        return self._print_seq(e.value, "{", "}")

    def _print_Mod(self, expr):
        if len(expr.args) > 2:
            return self._print_Function(expr)
        return super()._print_Mod(expr)

    def _print_Plus(self, expr):
        return super()._print_Add(s.Add(*expr.args, evaluate=False))

    def _print_Power(self, expr):
        return super()._print_Pow(s.Pow(*expr.args, evaluate=False))

    def _print_Times(self, expr):
        return super()._print_Mul(s.Mul(*expr.args, evaluate=False))

    def _print_Max(self, expr):
        return self._print_Function(expr)

    def _print_Min(self, expr):
        return self._print_Function(expr)

    @staticmethod
    def _print_ComplexInfinity(*args):
        return prettyForm("ComplexInfinity")

    def _helper_print_function(
        self, func, args, sort=False, func_name=None, delimiter=", ", elementwise=False
    ):
        if sort:
            args = sorted(args, key=s.utilities.default_sort_key)

        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        if func_name:
            if func_name in FunctionWrappersReverse and not issubclass(
                func, DefinedFunction
            ):
                func_name = FunctionWrappersReverse[func_name]
            prettyFunc = self._print(s.Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens(left="[", right="]"))

        if elementwise:
            if self._use_unicode:
                circ = pretty_atom("Modifier Letter Low Ring")
            else:
                circ = "."
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE, *stringPict.next(prettyFunc, circ)
            )

        prettyArgs = prettyForm(
            *self._print_seq(args, delimiter=delimiter).parens(left="[", right="]")
        )

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs)
        )

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform


def pretty_print(expr, **settings):
    pp = Printer9000(settings)

    return pp.doprint(expr)
