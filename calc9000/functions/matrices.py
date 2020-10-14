from calc9000.functions.core import *
from calc9000.functions.base import Conjugate
from calc9000.functions.list_funcs import do_list_add_mul

from functools import reduce
from sympy.matrices.common import NonInvertibleMatrixError, ShapeError


def isVector(m):
    if isinstance(m, List):
        return all(not isinstance(x, List) for x in m)


def toVectorList(m):
    """
    Converts A vector matrix into a single row list.
    """
    return do_list_add_mul(m)


def matrixToList(m):
    temp_list = []
    for row in range(m.rows):
        temp_list.append(do_list_add_mul(m.row(row)))
    return List(*temp_list)


class Dot(NormalFunction):

    tags = {
        'shap': 'Invalid matrix shapes for Dot.',
        'rect': 'Non-rectangular tensor encountered.'
    }

    @classmethod
    def do_dot(cls, a, b):
        v1 = isVector(a)
        v2 = isVector(b)

        try:
            if v1 and v2:
                # perform scalar dot product when both matrices are vectors
                return s.simplify(sum(List.create(a) * List.create(b)))
            if v1 or v2:
                # return a vector
                if v1:
                    return toVectorList(s.Matrix([a]) * s.Matrix(b))
                return toVectorList(s.Matrix(a) * s.Matrix(b))
            return matrixToList(s.Matrix(a) * s.Matrix(b))
        except (ShapeError, ValueError) as e:
            if isinstance(e, ValueError):
                if e.args[0].startswith('expecting list of lists') \
                        or e.args[0].startswith('mismatched dimensions'):
                    raise FunctionException('Dot::rect')
            # TODO: replace with more verbose info
            raise FunctionException('Dot::shap')

    @classmethod
    def exec(cls, *args):
        if not all(isinstance(x, iterables) for x in args):
            return None

        return reduce(cls.do_dot, args)


class Det(NormalFunction):
    """
    Det [m]
     Gives the Determinant of Square Matrix m.

    Equivalent to sympy.Matrix.det().
    """

    tags = {
        'sqr': 'Cannot find determinant of non-square matrix.'
    }

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            m = s.Matrix(x)
            if m.is_square:
                return m.det()
            raise FunctionException('Det::sqr')
        return None


class Inverse(NormalFunction):
    """
    Inverse [m]
     Gives the Inverse of Square Matrix m.

    Equivalent to sympy.Matrix.inv().
    """

    tags = {
        'sqr': 'Cannot find inverse of non-square matrix.',
        'val': 'Cannot find inverse of singular matrix.'
    }

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            try:
                m = s.Matrix(x)
                if m.is_square:
                    return matrixToList(m.inv())
                raise FunctionException('Inverse::sqr')
            except NonInvertibleMatrixError:
                raise FunctionException('Inverse::val')
        return None


class Transpose(NormalFunction):
    """
    Transpose [m]
     Gives the Transpose of Matrix m.

    Equivalent to sympy.Matrix.transpose().
    """

    # TODO: don't allow vector transpose

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            m = s.Matrix(x)
            return matrixToList(m.transpose())
        return None


class ConjugateTranspose(NormalFunction):
    """
    ConjugateTranspose [m]
     Gives the conjugate transpose of m.

    Equivalent to Conjugate[Transpose[m]].
    """

    @classmethod
    def exec(cls, x):
        if isinstance(x, iterables):
            return Transpose(Conjugate(x))
        return None


class Cross(NormalFunction):

    tags = {
        'dim': 'Cross product of vectors greater than 3 dimensions is not supported.'
    }

    @classmethod
    def exec(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], iterables) and len(args[0]) == 2:
                return List(args[0][1] * -1, args[0][0])
        elif len(args) == 2:
            if all(isinstance(x, iterables) for x in args):
                if len(args[0]) == len(args[1]) == 3:
                    return List.create(s.Matrix(args[0]).cross(s.Matrix(args[1])))
        raise FunctionException('Cross::dim')

    def _sympystr(self, printer=None):
        return 'Cross['.join(str(i) + ', ' for i in (printer.doprint(i) for i in self.args))[:-2] + ']'

