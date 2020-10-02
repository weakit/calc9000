from calc9000.functions.core import *
from calc9000.functions.base import *
from calc9000.functions.calculus import *
from calc9000.functions.expressions import *
from calc9000.functions.list_funcs import *
from calc9000.functions.matrices import *
from calc9000.functions.misc import *
from calc9000.functions.num_theory import *
from calc9000.functions.numbers import *
from calc9000.functions.random import *


r.refs.BuiltIns.update({
    k: v for k, v in globals().items()
    if isinstance(v, type) and issubclass(v, s.Function) and not issubclass(v, LazyFunction)
})
