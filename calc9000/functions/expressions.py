from calc9000.functions.core import *


class Factor(NormalFunction):
    """
    Factor [Expr, Modulus -> mod, Extension -> ext, GaussianIntegers -> bool}]
     Factors an Expression.

    Equivalent to sympy.factor().
    """

    op_spec = (
        {"Modulus": "modulus",
         "Extension": "extension",
         "GaussianIntegers": "gaussian"
         },
        None)

    param_spec = (1, 1)

    @classmethod
    def exec(cls, expr, **kws):
        return r_thread(s.factor, expr, **kws)


class Expand(NormalFunction):
    """
    Expand [Expr, Modulus -> mod]
     Expands an Expression.

    Equivalent to sympy.expand().
    """

    op_spec = (
        {"Modulus": "modulus", "Trig": "trig"},
        {"trig": False}
    )

    param_spec = (1, 1)

    @classmethod
    def exec(cls, expr, **kws):
        return r_thread(s.expand, expr, **kws)


class TrigExpand(NormalFunction):
    """
    TrigExpand [Expr]
     Expands only Trigonometric Functions.

    Equivalent to sympy.expand_trig().
    """

    @classmethod
    def exec(cls, expr):
        return r_thread(s.expand_trig, expr)


class ComplexExpand(NormalFunction):
    """
    ComplexExpand[expr]
     Expands expr assuming that all variables are real.

    ComplexExpand [expr, {x1, x2, …}]
     Expands expr assuming that variables matching any of the x are complex.

    Uses sympy.expand_complex().
    """

    @classmethod
    def exec(cls, x, complexes=()):
        def exp(expr):
            return s.refine(s.expand_complex(expr),
                            ands(s.Q.real(a) for a in expr.atoms(s.Symbol) if a not in complexes))

        if not isinstance(complexes, iterables):
            complexes = (complexes,)
        return thread(exp, x)


class Solve(NormalFunction):
    """
    Solve [expr, vars]
     Attempts to solve the system expr of equations or inequalities for the
     variables vars.

    Uses sympy.solve().
    """

    @classmethod
    def exec(cls, expr, v=None, dom=None):
        # TODO: fix (?)
        # if dom is None:
        #     dom = s.Complexes
        if not isinstance(expr, iterables):
            expr = List(expr)

        flat_expr = []

        # flatten ands
        for ex in expr:
            if isinstance(ex, s.And):
                for x in ex.args:
                    flat_expr.append(x)
            else:
                flat_expr.append(ex)

        solveset = []
        for ex in flat_expr:
            if isinstance(ex, s.Eq) and any(isinstance(x, iterables) for x in ex.args):
                solveset += list(thread(s.Eq, ex.args[0], ex.args[1]))
            else:
                solveset.append(ex)

        ret = s.solve(solveset, v, dict=True)

        return List(*[Rule.from_dict(x, head=List) for x in ret])


class Simplify(NormalFunction):
    """
    Simplify [expr]
     Attempts to simplify the expression expr.

    Equivalent to sympy.simplify().
    """

    @classmethod
    def exec(cls, expr, assum=None):
        if assum is not None:
            raise NotImplementedError("Assumptions not implemented.")
            # if isinstance(assum, iterables):
            #     for i in range(len(assum)):
            #         if isinstance(assum[i], s.core.py.relational.Relational):
            #             assum[i] = s.Q.is_true(assum[i])
            #
            #     assum = assumptions(assum)
            #     expr = thread(lambda x: s.refine(x, assum), expr)
        return r_thread(s.simplify, expr)


class Together(NormalFunction):
    @classmethod
    def exec(cls, expr):
        return thread(lambda x: s.simplify(s.together(x)), expr)


class Apart(NormalFunction):
    @classmethod
    def exec(cls, expr, x=None):
        return thread(s.apart, expr, x)


class Collect(NormalFunction):
    """
    Collect [expr, x]
     Collects together terms involving the same powers of objects matching x.

    Collect [expr, {x1, x2, …}]
     Collects together terms that involve the same powers of objects matching x1, x2, ….

    Collect [expr, var, h]
     Applies h to the expression that forms the coefficient of each term obtained.

    """

    tags = {
        'head': 'Invalid Function.'
    }

    @staticmethod
    def collect_func(expr, v, h):
        unevaluated_expr = s.collect(s.expand(expr), v, evaluate=False)
        expr = 0
        if h:
            if not isinstance(h, (s.Symbol, s.Function)):
                raise FunctionException('Collect::head')
            for c in unevaluated_expr:
                expr += Functions.call(str(h), unevaluated_expr[c]) * c
        else:
            for c in unevaluated_expr:
                expr += unevaluated_expr[c] * c
        return expr

    @classmethod
    def exec(cls, expr, v, h=None):
        if isinstance(expr, iterables):
            return List(*(cls.exec(x, v, h) for x in expr))
        return cls.collect_func(expr, v, h)


class Surd(NormalFunction):
    """
    Surd [x, n]
     Gives the real-valued nth root of x.

    Equivalent to sympy.real_root().
    """

    @classmethod
    def exec(cls, x, n):
        return thread(s.real_root, x, n)
