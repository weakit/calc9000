# `calc9000`


[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e47ea24859a6423b8f46e472ab27cb6d)](https://www.codacy.com/manual/weakit/calc9000?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=weakit/calc9000&amp;utm_campaign=Badge_Grade) 
[![codecov](https://codecov.io/gh/weakit/calc9000/branch/master/graph/badge.svg)](https://codecov.io/gh/weakit/calc9000)

`calc9000` aims to be a small and portable, general-purpose computer algebra system.

It is heavily based off of [Mathematica/WolframScript](https://www.wolfram.com/mathematica/).

`calc9000` internally uses [`sympy`](https://github.com/sympy/sympy) to perform most CAS functions.

## `installation + usage`

Python 3.9 as of yet is not supported. \
It is recommended to use a 3.8.x release to avoid any problems.

To install, simply clone the repo and run directly.

```shell script
$ git clone https://github.com/weakit/calc9000.git
$ cd calc9000
$ python -m pip install -r requirements.txt
```

`calc9000` is updated frequently, so please pull before running:

```shell script
$ git fetch 
$ git pull
```

To run do:
```shell script
$ python ./main.py
```

Syntax is almost identical to Mathematica, and most functions should just work.

---

This is a hobby project. A lot of planned functionality is missing, and although `calc9000` is perfectly usable, it is incomplete.

If you're looking for something more complete and robust, take a look at [mathics](https://github.com/mathics/Mathics).  
