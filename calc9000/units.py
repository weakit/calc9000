import sympy.physics.units as u

units = {n: x for n, x in vars(u).items() if isinstance(x, u.Quantity)}


def from_str_find_unit(s):
    if s in units:
        return units[s]
    return None

