import sympy as s
from calc9000 import references as r, larker
from calc9000.printer import pretty_print

parser = larker.parser


def process(input_text: str):
    if not input_text or input_text.isspace():
        r.refs.add_def("", "")
        return None
    out = parser.evaluate(input_text, r)
    # r.refs.add_def(input_text, out)
    return out


def process_pretty(input_text):
    raw = process(input_text)
    if raw is None:
        return None
    try:
        return pretty_print(raw)
    except TypeError:
        return raw


def current_line():
    return r.refs.Line


def previous_line():
    return r.refs.Line - 1
