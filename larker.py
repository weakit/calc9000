import expressions as ex
from lists import List, Rule
from lark import Lark, Transformer, Tree, Token


class AssignTransformer(Transformer):
    def plus(self, items):
        return ex.plus(items)

    def subtract(self, items):
        return ex.subtract(items)

    def times(self, items):
        return ex.times(items)

    def dot(self, items):
        return ex.dot(items)

    def positive(self, items):
        return ex.positive(items)

    def negative(self, items):
        return ex.negative(items)

    def divide(self, items):
        return ex.divide(items)

    def power(self, items):
        return ex.power(items)

    def factorial(self, items):
        return ex.factorial(items)

    def function(self, items):
        return ex.function(items)

    def list(self, items):
        return List(items)

    def rule(self, items):
        return Rule(*items)

    def relation(self, items):
        return ex.relations(items)

    def out(self, items):
        return ex.out(items)

    def INT(self, n):
        return ex.numeric(n)

    def FLOAT(self, n):
        return ex.numeric(n)

    def CNAME(self, n):
        return str(n)

    def symbol(self, n):
        return str(n[0])

    def RELATIONAL(self, n):
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

    def symbol(self, n):
        return ex.symbol(n[0])

    def handle(self, tree: Tree):
        if tree.data == "set":
            children = tree.children[:]
            for x in range(len(children) - 1):
                children[x] = assigner.transform(children[x])
            children[-1] = self.transform(children[-1])
            return ex.assign(children)
        elif tree.data == "unset":
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
