import platform
import converse as c
import lark


def print_info():
    print(f'--- calc9000 [{platform.python_implementation()} {platform.python_version()}]\n')


if __name__ == '__main__':
    print_info()
    try:
        while True:
            i = input(f'IN  {c.current_line()}: ')
            try:
                out = str(c.process_pretty(i))
            except KeyboardInterrupt as e:
                if isinstance(e, lark.exceptions.UnexpectedInput):
                    print(str(e)[:str(e).rindex('Expect')-2])
                else:
                    print(e)
                continue
            if '\n' in out:
                print(f'OUT {c.previous_line()}:')
                for line in out.splitlines():
                    print(f'\t{line}')
                print()
            else:
                print(f'OUT {c.previous_line()}: {out}\n')
    except KeyboardInterrupt:
        print('\n\nexiting.\n')
        exit(0)
