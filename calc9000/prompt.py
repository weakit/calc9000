import platform

try:
    from prompt_toolkit import *
    pft = print_formatted_text
except ImportError:
    print('Please install prompt_toolkit.')
    exit(-1)


# try:
#     from mathematica import MathematicaLexer
#     from mathematica.lexer import MToken
#     from prompt_toolkit.lexers import PygmentsLexer
#     from pygments.style import Style
#     from prompt_toolkit.styles.pygments import style_from_pygments_cls
#     # lex = PygmentsLexer(MathematicaLexer)
#     # style = style_from_pygments_cls(PromptStyle)
# except ImportError:
#     lex = None
#     style = None

lex = None
style = None

p = PromptSession(lexer=lex, style=style)


def print_startup():
    text = HTML(f'\n--- <b>calc9000</b> [running on {platform.python_implementation()} {platform.python_version()}] ')
    pft(text, end='')


def update_startup(v):
    pft(f'\b\b, using sympy {v}]\n')


def quit_prompt():
    pft(HTML('\n --- <i>Have a nice day!</i>\n'))
    exit(0)


def get_input(line):
    prompt_text = HTML(f'<green>In [<lime>{line}</lime>]:</green> ')
    return p.prompt(prompt_text, lexer=lex)


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


def handle_prompt(con):
    i = get_input(con.current_line())

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
    c.set_messenger(PromptMessenger())

    try:
        while True:
            handle_prompt(c)
    except KeyboardInterrupt:
        quit_prompt()
