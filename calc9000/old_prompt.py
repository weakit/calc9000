import platform
from sys import stderr

try:
    import colorama

    colorama.init()
    st, ed = colorama.Fore.RED, colorama.Fore.RESET
except ImportError:
    st, ed = "", ""


class StderrMessenger:
    @staticmethod
    def show(tag, m):
        print(f"{st}{tag}: {m}{ed}", file=stderr)


def _quit(*args):
    print("\n\n--- calc9000\n")
    exit(0)


def main():
    print(
        f"\n--- calc9000 [running on {platform.python_implementation()} {platform.python_version()}] ",
        end="",
    )

    import sympy as s

    from calc9000 import converse as c

    c.set_messenger(StderrMessenger())

    print(f"\b\b, using sympy {s.__version__}]\n")

    try:
        while True:

            i = input(f"IN  {c.current_line()}: ")

            try:
                out = c.process_pretty(i)
                if out is None:
                    print("")
                    continue
            except KeyboardInterrupt:
                continue

            if "\n" in out:
                print(f"OUT {c.previous_line()}:")
                for line in out.splitlines():
                    print(f"\t{line}")
                print()

            else:
                print(f"OUT {c.previous_line()}: {out}\n")
    except KeyboardInterrupt:
        _quit()
