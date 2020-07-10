import forge as op
# import sympy as s
# from datatypes import List, Rule
from lark import Lark, Transformer, Tree, Token


class AssignTransformer(Transformer):
    # @staticmethod
    # def plus(items):
    #     return op.plus(items)
    #
    # @staticmethod
    # def subtract(items):
    #     return op.subtract(items)
    #
    # @staticmethod
    # def times(items):
    #     return op.times(items)
    #
    # @staticmethod
    # def dot(items):
    #     return op.dot(items)
    #
    # @staticmethod
    # def positive(items):
    #     return op.positive(items)
    #
    # @staticmethod
    # def negative(items):
    #     return op.negative(items)
    #
    # @staticmethod
    # def divide(items):
    #     return op.divide(items)
    #
    # @staticmethod
    # def power(items):
    #     return op.power(items)
    #
    # @staticmethod
    # def factorial(items):
    #     return op.factorial(items)
    #
    # @staticmethod
    # def function(items):
    #     return op.unset_function(items)
    #
    # @staticmethod
    # def list(items):
    #     return List(*items)
    #
    # @staticmethod
    # def rule(items):
    #     return Rule(*items)
    #
    # @staticmethod
    # def rule_(items):
    #     return Rule(*items)
    #
    # @staticmethod
    # def relation(items):
    #     return op.relations(items)
    #
    # @staticmethod
    # def and_(items):
    #     return op.And(items)
    #
    # @staticmethod
    # def out(items):
    #     return op.out(items)
    #
    # @staticmethod
    # def part(items):
    #     return op.part(items)
    #
    # @staticmethod
    # def replace(items):
    #     return op.replace(items)

    @staticmethod
    def INT(n):
        return op.numeric(n)

    @staticmethod
    def FLOAT(n):
        return op.numeric(n)

    @staticmethod
    def CNAME(n):
        return str(n)

    # @staticmethod
    # def ESCAPED_STRING(n):
    #     return str(n)[1:-1]

    # @staticmethod
    # def symbol(n):
    #     return s.Symbol(str(n[0]))

    # @staticmethod
    # def RELATIONAL(n):
    #     return str(n)

    def transform(self, tree):
        if isinstance(tree, Token):
            return getattr(self, tree.type)(tree)
        return super().transform(tree)


class DelayedAssignTransformer(AssignTransformer):
    def __init__(self, funcs):
        self.funcs = funcs
        super().__init__()

    # @staticmethod
    # def symbol(n):
    #     return s.Symbol(str(n[0]))
    #
    # def function(self, items):
    #     self.funcs.append(items[0])
    #     return op.unset_function(items)


class SymbolTransformer(AssignTransformer):
    def __init__(self):
        super().__init__()
        with open("parser.lark", 'r') as f:
            lark = f.read()
        self.parser = Lark(lark, start='start', parser="lalr")

    # @staticmethod
    # def symbol(n):
    #     return op.symbol(n[0])
    #
    # @staticmethod
    # def function(items):
    #     return op.function(items)

    def handle(self, tree: Tree):
        if tree.data == "set":
            children = tree.children[:]
            for x in range(len(children) - 1):
                children[x] = assigner.transform(children[x])
            children[-1] = self.transform(children[-1])
            return op.assign(children)
        if tree.data == "unset":
            return op.unset(assigner.transform(tree.children[0]))
        if tree.data == "set_delayed":
            f = []
            transformer = DelayedAssignTransformer(f)
            children = tree.children[:]
            for x in range(len(children)):
                children[x] = transformer.transform(children[x])
            return op.delayed(children, list(transformer.funcs))

    def evaluate(self, t):
        parsed = self.parser.parse(t)
        if isinstance(parsed, Tree):
            if parsed.data in ["set", "unset", "set_delayed"]:
                return self.handle(parsed)
            return op.operate(self.transform(parsed))
        return op.operate(parsed)


if __name__ == "larker":
    assigner = AssignTransformer()
    parser = SymbolTransformer()
