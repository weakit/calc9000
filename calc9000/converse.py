import sympy as s
from calc9000 import references as r, larker
from calc9000.printer import pretty_print


parser = larker.parser
refs = r.refs


def process(input_text: str):

    if not input_text or input_text.isspace():
        refs.add_def('', '')
        return None

    out = parser.evaluate(input_text)

    if isinstance(out, (r.NoOutput,)):
        refs.add_def(input_text, out.value)
        out = None
    else:
        refs.add_def(input_text, out)
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
    return refs.Line


def previous_line():
    return refs.Line - 1


def set_messenger(m):
    if not hasattr(m, 'show'):
        pass  # Raise Exception
    r.refs.Messenger = m
