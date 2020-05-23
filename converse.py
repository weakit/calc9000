import sympy
import larker
import references as r


parser = larker.parser


def process(input_text: str):
    if not input_text or input_text.isspace():
        r.refs.add_def("", "")
        return None
    out = parser.parse(input_text)
    r.refs.add_def(input_text, out)
    return out


def process_pretty(input_text):
    raw = process(input_text)
    return sympy.pretty(raw)


def current_line():
    return r.refs.Line


def previous_line():
    return r.refs.Line - 1
