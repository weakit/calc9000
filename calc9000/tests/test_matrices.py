from . import p


def test_dot():
    assert p("{a, b, c} . {x, y, z}") == p("a*x + b*y + c*z")
    assert p("{{a, b}, {c, d}} . {x, y}") == p("{a*x + b*y, c*x + d*y}")
    assert p("{x, y} . {{a, b}, {c, d}}") == p("{a*x + c*y, b*x + d*y}")
    assert p("{x, y} . {{a, b}, {c, d}} . {r, s}") == p("r*(a*x + c*y) + s*(b*x + d*y)")
    assert p("{{a, b}, {c, d}} . {{r, s}, {t, u}}") == p(
        "{{a*r + b*t, a*s + b*u}, {c*r + d*t, c*s + d*u}} "
    )
    assert p("p = {{1, 1, 0}, {0, 1, 1}, {0, 0, 1}}; Dot[p, p, p]") == p(
        "{{1,3,3},{0,1,3},{0,0,1}}"
    )
    assert p("{{1, 2}, {3, 4}, {5, 6}}.{1, 1}") == p("{3, 7, 11}")
    assert p("{{1, 2}, {3, 4}, {5, 6}}.{{1}, {1}}") == p("{{3},{7},{11}}")
    assert p("{1, 1, 1}.{{1, 2}, {3, 4}, {5, 6}}") == p("{9,12}")
    assert p("{{1, 1, 1}}.{{1, 2}, {3, 4}, {5, 6}}") == p("{{9,12}}")
    assert p("{{1}, {2}, {3}}.{{4, 5, 6}}") == p("{{4,5,6},{8,10,12},{12,15,18}}")
    assert p(
        "{2,0}.{{{7,8,1,3},{4,4,1,8},{1,1,9,8}},{{1,5,9,8},{5,5,0,0},{9,1,4,3}}}"
    ) == p("{{14,16,2,6},{8,8,2,16},{2,2,18,16}}")


def test_det():
    assert p("Det[{{a, b}, {c, d}}]") == p("a*d - b*c")
    assert p("Det[{{a, b, c}, {d, e, f}, {g, h, i}}]") == p(
        "-(c*e*g) + b*f*g + c*d*h - a*f*h - b*d*i + a*e*i"
    )
    assert p("Det[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]") == p("0")


def test_inv():
    assert p("Inverse[{{1, 2}, {3, -6}}]") == p("{{1/2, 1/6}, {1/4, -1/12}}")
    assert p("Inverse[{{u, v}, {v, u}}]") == p(
        "{{u/(u^2 - v^2), -(v/(u^2 - v^2))}, {-(v/(u^2 - v^2)), u/(u^2 - v^2)}}"
    )
    assert p("m = {{a, b}, {c, d}}; Simplify[Inverse[m].m]") == p("{{1, 0}, {0, 1}}")


def test_transpose():
    assert p("Transpose[{{a, b, c}, {x, y, z}}]") == p("{{a, x}, {b, y}, {c, z}}")
    assert p("Transpose[{{a, x}, {b, y}, {c, z}}]") == p("{{a, b, c}, {x, y, z}}")


def test_conjugate_transpose():
    assert p("ConjugateTranspose[{{1, 2 I, 3}, {3 + 4 I, 5, I}}]") == p(
        "{{1, 3 - 4*I}, {-2*I, 5}, {3, -I}} "
    )
    assert p("ConjugateTranspose[{{a, b}, {c, d}}]") == p(
        "{{Conjugate[a], Conjugate[c]}, {Conjugate[b], Conjugate[d]}}"
    )


def test_cross():
    assert p("Cross[{a, b, c}, {x, y, z}]") == p("{b*z - c*y, -a*z + c*x, a*y - b*x}")
    assert p("Cross[{x, y}]") == p("{-y, x}")
