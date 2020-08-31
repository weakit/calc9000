import sympy as s
from calc9000.functions import Dot, Cross
from calc9000.datatypes import List
from sympy.printing.pretty.pretty import PrettyPrinter, prettyForm, sstr, \
    precedence_traditional, PRECEDENCE


class Printer9000(PrettyPrinter):
    def _print_Float(self, e):
        e = s.Float(e, precision=e._prec-12)
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

    def _print_Rule(self, e):
        return self._print_Implies(List(e.lhs, e.rhs), altchar='->')

    def _print_List(self, e):
        return self._print_seq(e.value, '{', '}')


def pretty_print(expr, **settings):
    pp = Printer9000(settings)

    return pp.doprint(expr)
