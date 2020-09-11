import sympy as s
from calc9000.functions import Dot, Cross, Limit, Subs
# from calc9000.datatypes import List
from calc9000.references import FunctionWrappersReverse
from sympy.printing.pretty.pretty import PrettyPrinter, prettyForm, sstr, \
    precedence_traditional, PRECEDENCE, pretty_atom, stringPict


class Printer9000(PrettyPrinter):

    def _print_Float(self, e):
        # TODO: fix float printing in List
        e = s.Float(e, precision=max(e._prec-13, 1))
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
            return self._print_seq(e.args, None, None, '.',
                                   parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])
        return super()._print_Dot(e)

    def _print_Cross(self, e):
        if isinstance(e, Cross):
            return self._print_seq(e.args, None, None, '×',
                                   parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])
        return super()._print_Cross(e)

    # def _print_Rule(self, e):
    #     TODO: Proper rule printing
    #     return self._print_Implies(List(
    #         self._print_seq(e.lhs),
    #         self._print_seq(e.rhs)
    #     ), altchar='->')

    def _print_Limit(self, l):
        if isinstance(l, Limit):
            return super()._print_Function(l)
        return super()._print_Limit(l)

    def _print_List(self, e):
        return self._print_seq(e.value, '{', '}')

    def _print_Subs(self, e):
        if isinstance(e, Subs):
            return super()._print_Function(e)
        return super()._print_Subs(e)

    def _print_Mod(self, expr):
        if len(expr.args) > 2:
            return self._print_Function(expr)
        return super()._print_Mod(expr)

    def _helper_print_function(self, func, args, sort=False, func_name=None, delimiter=', ', elementwise=False):
        if sort:
            args = sorted(args, key=s.utilities.default_sort_key)

        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        if func_name:
            if func_name in FunctionWrappersReverse:
                func_name = FunctionWrappersReverse[func_name]
            prettyFunc = self._print(s.Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens(left='[', right=']'))

        if elementwise:
            if self._use_unicode:
                circ = pretty_atom('Modifier Letter Low Ring')
            else:
                circ = '.'
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE,
                *stringPict.next(prettyFunc, circ)
            )

        prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens(left='[', right=']'))

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform


def pretty_print(expr, **settings):
    pp = Printer9000(settings)

    return pp.doprint(expr)
