import html
import platform

try:
    from prompt_toolkit import *

    pft = print_formatted_text
except ImportError:
    raise EnvironmentError

lex = None
style = None
e = html.escape


def setup_lexer(builtins=None):
    global lex, style
    try:
        import pygments.token as tk
        from prompt_toolkit.lexers import PygmentsLexer
        from prompt_toolkit.styles import style_from_pygments_cls
        from pygments.lexer import RegexLexer, bygroups
        from pygments.style import Style

        # stupid, but works
        builtins_regex = (
            r"\b(" + builtins[0] + "".join([f"|{x}" for x in builtins[1:]]) + r")\b"
        )

        class SimpleLexer(RegexLexer):
            tokens = {
                "root": [
                    (r"\s+", tk.Text),
                    (r"\(\*[^(\*\))]*(\*\))?", tk.Comment),
                    # (builtins_regex, tk.Keyword),
                    # (r'((\d+)?(\.\d+)|(\d+)\.?)(`\d+)?', tk.Number),
                    (r'"[^"]*"?', tk.String),
                ]
            }

        lex = PygmentsLexer(SimpleLexer)
    except ImportError:
        pass


def check_console():
    from prompt_toolkit.output.win32 import (
        NoConsoleScreenBufferError as win_err,
    )

    try:
        pft("", end="")
    except win_err:
        raise EnvironmentError("Fallback", -1)


def print_startup():
    text = HTML(
        f"\n--- <b>calc9000</b> [running on {platform.python_implementation()} {platform.python_version()}] "
    )
    pft(text, end="")


def update_startup(v):
    pft(f"\b\b, using sympy {v}]\n")


def quit_prompt():
    pft(HTML("\n--- <i>Have a nice day!</i>\n"))
    exit(0)


def setup_prompt_sessions(con):
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    bindings = KeyBindings()

    @bindings.add(Keys.ControlZ)
    def _(event):
        pft()
        quit_prompt()

    @bindings.add(Keys.ControlO)
    def _(event):
        pass

    def prompt_continuation(width, line_number, is_soft_wrap):
        return HTML("<green>" + " " * (width - 5) + "...: </green>")

    class LimitedWordCompleter(WordCompleter):
        @staticmethod
        def denier():
            return
            yield

        def get_completions(self, document, complete_event):
            if len(document.get_word_before_cursor()) > 2:
                return super().get_completions(document, complete_event)
            return self.denier()

    builtins_completer = LimitedWordCompleter(con.get_builtins())
    session = PromptSession(
        lexer=lex,
        style=style,
        completer=builtins_completer,
        complete_in_thread=True,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        prompt_continuation=prompt_continuation,
    )
    return session


def get_input(line, prompt_session):
    prompt_text = HTML(f"<green>In [<lime>{line}</lime>]:</green> ")
    return prompt_session.prompt(prompt_text, lexer=lex)


def display_output(out, line):
    if out is None:
        pft()
        return

    pft(HTML(f"<FireBrick>Out[<red>{line}</red>]: </FireBrick>"), end="")

    if "\n" in out:
        # handle multiline output
        pft()
        for line in out.splitlines():
            pft(f"\t{line}")
        pft()

    else:
        pft(out, end="\n\n")


class PromptMessenger:
    @staticmethod
    def show(tag, message):
        pft(HTML(f"<red>{e(str(tag))}: {e(str(message))}</red>"))


def handle_prompt(con, prompt_session):
    i = get_input(con.current_line(), prompt_session)

    try:
        out = con.process_pretty(i)
    except KeyboardInterrupt:
        return

    display_output(out, con.previous_line())


def main():
    check_console()
    print_startup()
    import sympy as s

    from calc9000 import converse as c

    update_startup(s.__version__)
    setup_lexer(c.get_builtins())
    p = setup_prompt_sessions(c)
    c.set_messenger(PromptMessenger())

    try:
        while True:
            handle_prompt(c, p)
    except KeyboardInterrupt:
        quit_prompt()
