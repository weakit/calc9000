if __name__ == "__main__":
    try:
        from calc9000 import prompt

        prompt.main()
    except EnvironmentError as e:
        if e.args != ("Fallback", -1):
            raise e
        from calc9000 import old_prompt

        old_prompt.main()
