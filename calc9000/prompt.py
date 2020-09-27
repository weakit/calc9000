import platform

try:
    from prompt_toolkit import *

    pft = print_formatted_text
except ImportError:
    print('Please install prompt_toolkit.')
    exit(-1)

lex = None
style = None


def setup_lexer(builtins: None):
    global lex, style
    try:
        from pygments.style import Style
        from pygments.lexer import RegexLexer, bygroups
        import pygments.token as tk
        from prompt_toolkit.lexers import PygmentsLexer
        from prompt_toolkit.styles import style_from_pygments_cls

        # stupid, but works
        builtins_regex = \
            r'\b(' + builtins[0] + ''.join([f'|{x}' for x in builtins[1:]]) + r')\b'

        class SimpleLexer(RegexLexer):
            tokens = {
                'root': [
                    (r'\s+', tk.Text),
                    (r'\(\*[^(\*\))]*(\*\))?', tk.Comment),
                    # (builtins_regex, tk.Keyword),
                    # (r'((\d+)?(\.\d+)|(\d+)\.?)(`\d+)?', tk.Number),
                    (r'"[^"]*"?', tk.String),
                ]
            }

        lex = PygmentsLexer(SimpleLexer)
    except ImportError:
        pass


def print_startup():
    text = HTML(f'\n--- <b>calc9000</b> [running on {platform.python_implementation()} {platform.python_version()}] ')
    pft(text, end='')


def update_startup(v):
    pft(f'\b\b, using sympy {v}]\n')


def quit_prompt():
    pft(HTML('\n--- <i>Have a nice day!</i>\n'))
    exit(0)


def setup_prompt_sessions(con):
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

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
    return PromptSession(
        lexer=lex,
        style=style,
        completer=builtins_completer,
        complete_in_thread=True,
        auto_suggest=AutoSuggestFromHistory(),
    )


def get_input(line, prompt_session):
    prompt_text = HTML(f'<green>In [<lime>{line}</lime>]:</green> ')
    return prompt_session.prompt(prompt_text, lexer=lex)


def display_output(out, line):
    if out is None:
        pft()
        return

    pft(HTML(f'<FireBrick>Out[<red>{line}</red>]: </FireBrick>'), end='')

    if '\n' in out:
        # handle multiline output
        pft()
        for line in out.splitlines():
            pft(f'\t{line}')
        pft()

    else:
        pft(out, end='\n\n')


class PromptMessenger:
    @staticmethod
    def show(tag, message):
        pft(HTML(f'<red>{tag}: {message}</red>'))


def handle_prompt(con, prompt_session):
    i = get_input(con.current_line(), prompt_session)

    try:
        out = con.process_pretty(i)
    except KeyboardInterrupt:
        return

    display_output(out, con.previous_line())


def main():
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
