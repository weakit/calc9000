from calc9000 import forge as op
from lark import Lark, Transformer, Token
from calc9000.custom import String, CustomException


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
           | span

?assign: replace "=" assign -> set
       | replace ("=" "."|"=.") -> unset
       | replace ":=" assign -> set_delayed
       | replace

// makeshift span, tree is altered after parsing
// does not work the same way as mathematica, but works decently enough
// cannot assign, or use in other ways

span: expression? (SPAN expression?)+

SPAN: ";;"

?replace: replace "/." postfix
        | postfix

?postfix: logic "//" postfix
        | logic

?logic: rule
      | logic ("&&" rule)+ -> and_
      | logic ("||" rule)+ -> or_

?rule: relation "->" rule
     | relation

// rule_: expression "->" expression
//      | "{" rule_ ("," rule)* "}" -> list

?relation: addsub
         | relation (RELATIONAL addsub)+

RELATIONAL: "=="
          | "!="
          | ">="
          | "<="
          | ">"
          | "<"

?addsub: muldiv
       | addsub ("+" muldiv)+ -> plus
       | addsub "-" muldiv -> subtract

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
        | atom "[" "[" (compound_statement)? ("," compound_statement)* "]" "]" -> part

function: atom "[" (compound_statement)? ("," compound_statement)* "]"
// TODO | function "[" (expression)? ("," expression)* "]"

?unary: factorial
      | atom

?atom: "(" compound_statement ")"
     | execute
     | list
     | numeric
     | symbol
     | string
     | out
     | tag

tag: symbol "::" symbol

list: "{" (compound_statement ("," compound_statement )*)? "}"

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
        return String(n[1:-1])

    def transform(self, tree):
        if isinstance(tree, Token):
            return getattr(self, tree.type)(tree)
        return super().transform(tree)


class SymbolTransformer(AssignTransformer):
    def __init__(self):
        super().__init__()
        self.parser = Lark(grammar, start='start', parser="lalr")

    def evaluate(self, t):
        parsed = self.parser.parse(t)
        result = op.operate(self.transform(parsed))
        return result


if 'larker' in __name__:
    assigner = AssignTransformer()
    parser = SymbolTransformer()
