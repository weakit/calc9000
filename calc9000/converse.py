from lark.exceptions import LarkError

from calc9000 import larker
from calc9000 import references as r
from calc9000.custom import SpecialOutput, Tag
from calc9000.printer import pretty_print

parser = larker.parser
refs = r.refs


def _pre_process(s: str):
    # remove newlines
    s = s.replace("\n", "")

    # remove comments
    bef, aft = "", s
    while "(*" in aft or "*)" in aft:
        start = aft.find("(*")
        end = aft.find("*)")
        if start >= 0:
            if end == -1:
                raise SyntaxError("Unterminated comment.")
            bef += aft[:start]
            aft = aft[end + 2 :]
            continue
        raise SyntaxError("Unexpected comment terminator. *).")
    return bef + aft


def process(input_text: str):

    try:
        input_text = _pre_process(input_text)
    except SyntaxError as e:
        refs.add_message(Tag("Synatx::err"), str(e))
        return None

    if not input_text or input_text.isspace():
        refs.add_def("", "")
        return None

    try:
        out = parser.evaluate(input_text)
    except (LarkError, SyntaxError) as e:
        e = "".join((x + "\n\t" for x in str(e).split("\n")[:4] if x)).rstrip("\n\t")
        refs.add_message(Tag("Synatx::err"), e)
        return None

    if isinstance(out, SpecialOutput):
        refs.add_def(input_text, out.value)
        return out.value_to_print()

    refs.add_def(input_text, out)
    return out


def process_pretty(input_text):
    raw = process(input_text)
    if raw is None:
        return None
    try:
        return pretty_print(raw)
    except TypeError:
        if hasattr(raw, "pretty"):
            return raw.pretty()
        return raw


def current_line():
    return refs.Line


def previous_line():
    return refs.Line - 1


def set_messenger(m):
    if not hasattr(m, "show"):
        pass  # Raise Exception
    r.refs.Messenger = m


def get_builtins():
    return (
        list(r.refs.BuiltIns.keys())
        + list(r.refs.Constants.Dict.keys())
        + list(r.refs.Protected.Dict.keys())
    )
