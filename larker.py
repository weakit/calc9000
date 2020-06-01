import expressions as ex
from lists import List, Rule
from lark import Lark, Transformer, Tree, Token


class AssignTransformer(Transformer):
    @staticmethod
    def plus(items):
        return ex.plus(items)

    @staticmethod
    def subtract(items):
        return ex.subtract(items)

    @staticmethod
    def times(items):
        return ex.times(items)

    @staticmethod
    def dot(items):
        return ex.dot(items)

    @staticmethod
    def positive(items):
        return ex.positive(items)

    @staticmethod
    def negative(items):
        return ex.negative(items)

    @staticmethod
    def divide(items):
        return ex.divide(items)

    @staticmethod
    def power(items):
        return ex.power(items)

    @staticmethod
    def factorial(items):
        return ex.factorial(items)

    @staticmethod
    def function(items):
        return ex.function(items)

    @staticmethod
    def list(items):
        return List(items)

    @staticmethod
    def rule(items):
        return Rule(*items)

    @staticmethod
    def rule_(items):
        return Rule(*items)

    @staticmethod
    def relation(items):
        return ex.relations(items)

    @staticmethod
    def and_(items):
        return ex.And(items)

    @staticmethod
    def out(items):
        return ex.out(items)

    @staticmethod
    def part(items):
        return ex.part(items)

    @staticmethod
    def replace(items):
        return ex.replace(items)

    @staticmethod
    def INT(n):
        return ex.numeric(n)

    @staticmethod
    def FLOAT(n):
        return ex.numeric(n)

    @staticmethod
    def CNAME(n):
        return str(n)

    @staticmethod
    def ESCAPED_STRING(n):
        return str(n)[1:-1]

    @staticmethod
    def symbol(n):
        return str(n[0])

    @staticmethod
    def RELATIONAL(n):
        return str(n)

    def transform(self, tree):
        if isinstance(tree, Token):
            return getattr(self, tree.type)(tree)
        return super().transform(tree)


class SymbolTransformer(AssignTransformer):
    def __init__(self):
        super().__init__()
        with open("parser.lark", 'r') as f:
            lark = f.read()
        self.parser = Lark(lark, start='start', parser="lalr")

    @staticmethod
    def symbol(n):
        return ex.symbol(n[0])

    def handle(self, tree: Tree):
        if tree.data == "set":
            children = tree.children[:]
            for x in range(len(children) - 1):
                children[x] = assigner.transform(children[x])
            children[-1] = self.transform(children[-1])
            return ex.assign(children)
        if tree.data == "unset":
            return ex.unset(assigner.transform(tree.children[0]))

    def evaluate(self, t):
        parsed = self.parser.parse(t)
        if isinstance(parsed, Tree):
            if parsed.data in ["set", "unset"]:
                return self.handle(parsed)
            return self.transform(parsed)
        return parsed


if __name__ == "larker":
    assigner = AssignTransformer()
    parser = SymbolTransformer()
