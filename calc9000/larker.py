from calc9000 import forge as op
from lark import Lark, Transformer, Token


grammar = """
// calc9000 Parser Grammar

?start: compound_statement

?compound_statement: statement+

?statement: expression
          | expression ";" -> semicolon_statement

// set: (expression "=")+ expression
// set_delayed: expression ":=" expression
// unset: expression "=" "."

?expression: assign

?assign: replace "=" assign -> set
       | replace ("=" "."|"=.") -> unset
       | replace ":=" assign -> set_delayed
       | replace

?replace: replace "/." logic
        | logic

?logic: rule
      | "!" logic -> not_
      | logic ("&&" rule)+ -> and_
      | logic ("||" rule)+ -> or_

?rule: relation "->" rule
     | relation

// rule_: expression "->" expression
//      | "{" rule_ ("," rule)* "}" -> list

?relation: addsub
         | relation (RELATIONAL addsub)+

RELATIONAL: "=="
          | ">"
          | "<"
          | ">="
          | "<="

?addsub: muldiv
       | addsub ("+" muldiv)+ -> plus
       | addsub ("-" muldiv)+ -> subtract

?muldiv: prefix
       | muldiv "/" prefix -> divide
       | muldiv ("*" prefix)+ -> times
       | muldiv power -> times
       | muldiv "." prefix -> dot

?prefix: power
       | "+" prefix -> positive
       | "-" prefix -> negative

?power: unary
      | unary ("^" prefix)+

?execute: function
        | atom "[" "[" (expression)? ("," expression)* "]" "]" -> part

function: atom "[" (compound_statement)? ("," compound_statement)* "]"
// TODO | function "[" (expression)? ("," expression)* "]"

?unary: factorial
      | atom

?atom: "(" expression ")"
     | execute
     | list
     | numeric
     | symbol
     | string
     | out


list: "{" (expression ("," expression)*)? "}"

?assign_atom: function
            | assign_list
            | numeric
            | CNAME

assign_list: "{" (CNAME|assign_list) ("," (CNAME|assign_list))* "}"

factorial: unary "!"

?numeric: FLOAT
        | INT

symbol: CNAME

?string: ESCAPED_STRING

out: OUT+
   | OUT (INT)

OUT: "%"

%import common.CNAME
%import common.ESCAPED_STRING
%import common.INT
%import common.FLOAT
%import common.WS_INLINE
%ignore WS_INLINE
"""


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
        return op.float_(n)

    @staticmethod
    def CNAME(n):
        return str(n)

    @staticmethod
    def ESCAPED_STRING(n):
        return str(n)[1:-1]

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


# class DelayedAssignTransformer(AssignTransformer):
#     def __init__(self, funcs):
#         self.funcs = funcs
#         super().__init__()
#
#     @staticmethod
#     def symbol(n):
#         return s.Symbol(str(n[0]))
#
#     def function(self, items):
#         self.funcs.append(items[0])
#         return op.unset_function(items)


class SymbolTransformer(AssignTransformer):
    def __init__(self):
        super().__init__()
        self.parser = Lark(grammar, start='start', parser="lalr")

    # @staticmethod
    # def symbol(n):
    #     return op.symbol(n[0])
    #
    # @staticmethod
    # def function(items):
    #     return op.function(items)

    # def handle(self, tree: Tree):
    #     if tree.data == "set":
    #         children = tree.children[:]
    #         for x in range(len(children) - 1):
    #             children[x] = assigner.transform(children[x])
    #         children[-1] = self.transform(children[-1])
    #         return op.assign(children)
    #     if tree.data == "unset":
    #         return op.unset(assigner.transform(tree.children[0]))
    #     if tree.data == "set_delayed":
    #         f = []
    #         transformer = DelayedAssignTransformer(f)
    #         children = tree.children[:]
    #         for x in range(len(children)):
    #             children[x] = transformer.transform(children[x])
    #         return op.delayed(children, list(transformer.funcs))

    def evaluate(self, t):
        parsed = self.parser.parse(t)
        result = op.operate(self.transform(parsed))
        return result


if 'larker' in __name__:
    assigner = AssignTransformer()
    parser = SymbolTransformer()
