import sympy as s
from sympy.printing.pretty.pretty import PrettyPrinter, prettyForm, sstr


class OverSimplifier(PrettyPrinter):
    def _print_Float(self, e):
        e = s.Float(e, precision=e._prec-12)
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))


def pretty_print(expr, **settings):
    pp = OverSimplifier(settings)

    return pp.doprint(expr)
