import expressions as ex
from lists import List, Rule
from lark import Lark, Transformer, Tree


class Parser(Transformer):
    def __init__(self):
        super().__init__()
        with open("parser.lark", 'r') as f:
            lark = f.read()
        self.parser = Lark(lark, start='start', parser="lalr")

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

    def assign_list(self, items):
        return tuple(items)

    def symbolic_list(self, items):
        return List(items)

    def relation(self, items):
        return ex.relations(items)

    def set(self, items):
        return ex.assign(items)

    def unset(self, items):
        return ex.unset(items)

    def out(self, items):
        return ex.out(items)

    def INT(self, n):
        return ex.numeric(n)

    def FLOAT(self, n):
        return ex.numeric(n)

    def CNAME(self, n):
        return str(n)

    def symbol(self, n):
        return ex.symbol(n[0])

    def RELATIONAL(self, n):
        return str(n)

    def parse(self, t):
        parsed = self.parser.parse(t)
        if isinstance(parsed, Tree):
            return self.transform(parsed)
        return parsed


if __name__ == "larker":
    parser = Parser()
