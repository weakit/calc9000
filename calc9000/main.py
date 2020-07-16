import platform


if __name__.endswith('main'):
    print(f'\n--- calc9000 [running on {platform.python_implementation()} {platform.python_version()}] ', end='')
    from calc9000 import converse as c
    import lark
    print(f'\b\b, using sympy {c.s.__version__}]\n')
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
        print('\n\n--- Have a nice day!\n')
        exit(0)
