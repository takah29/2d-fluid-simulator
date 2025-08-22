import taichi as ti

from differentiation import diff_x, diff_y, fdiff_x, fdiff_y, sample


@ti.func
def advect(vc: ti.template(), phi: ti.template(), i: int, j: int, dx: float) -> float:  # type: ignore[valid-type]
    """Central Differencing."""
    return vc[i, j].x * diff_x(phi, i, j, dx) + vc[i, j].y * diff_y(phi, i, j, dx)


@ti.func
def advect_upwind(vc: ti.template(), phi: ti.template(), i: int, j: int, dx: float) -> float:  # type: ignore[valid-type]
    """Upwind differencing.

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#tac8e468
    """
    k = i if vc[i, j].x < 0.0 else i - 1
    a = vc[i, j].x * fdiff_x(phi, k, j, dx)

    k = j if vc[i, j].y < 0.0 else j - 1
    b = vc[i, j].y * fdiff_y(phi, i, k, dx)

    return a + b


@ti.func
def advect_kk_scheme(vc: ti.template(), phi: ti.template(), i: int, j: int, dx: float) -> float:  # type: ignore[valid-type]
    """Kawamura-Kuwabara Scheme.

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#y2dbc484
    """
    coef = [-2, 10, -9, 2, -1]
    v = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])

    v = ti.Vector(coef) if vc[i, j].x < 0 else -ti.Vector(coef[::-1])
    mx = ti.Matrix.cols(
        [
            sample(phi, i + 2, j),
            sample(phi, i + 1, j),
            sample(phi, i, j),
            sample(phi, i - 1, j),
            sample(phi, i - 2, j),
        ]
    )
    a = mx @ v / (6 * dx)

    v = ti.Vector(coef) if vc[i, j].y < 0 else -ti.Vector(coef[::-1])
    my = ti.Matrix.cols(
        [
            sample(phi, i, j + 2),
            sample(phi, i, j + 1),
            sample(phi, i, j),
            sample(phi, i, j - 1),
            sample(phi, i, j - 2),
        ]
    )
    b = my @ v / (6 * dx)

    return vc[i, j].x * a + vc[i, j].y * b
