from lark.exceptions import LarkError
from calc9000 import references as r, larker
from calc9000.printer import pretty_print
from calc9000.datatypes import Tag


parser = larker.parser
refs = r.refs


def process(input_text: str):

    if not input_text or input_text.isspace():
        refs.add_def('', '')
        return None

    try:
        out = parser.evaluate(input_text)
    except (LarkError, SyntaxError) as e:
        e = ''.join((x + '\n\t' for x in str(e).split('\n')[:4] if x)).rstrip('\n')
        refs.add_message(Tag('Synatx::err'), e)
        refs.add_def(input_text, None)
        return None

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
        if hasattr(raw, 'pretty'):
            return raw.pretty()
        return raw


def current_line():
    return refs.Line


def previous_line():
    return refs.Line - 1


def set_messenger(m):
    if not hasattr(m, 'show'):
        pass  # Raise Exception
    r.refs.Messenger = m
