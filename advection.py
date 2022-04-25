import taichi as ti

from differentiation import (
    sample,
    diff_x,
    diff_y,
    fdiff_x,
    fdiff_y,
    bdiff_x,
    bdiff_y,
    diff2_x,
    diff2_y,
    diff3_x,
    diff3_y,
)


@ti.func
def advect(vc, phi, i, j):
    """Central Differencing"""
    return vc[i, j].x * diff_x(phi, i, j) + vc[i, j].y * diff_y(phi, i, j)


@ti.func
def advect_upwind(vc, phi, i, j):
    """Upwind differencing

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#tac8e468
    """
    k = i if vc[i, j].x < 0.0 else i - 1
    a = vc[i, j].x * fdiff_x(phi, k, j)

    k = j if vc[i, j].y < 0.0 else j - 1
    b = vc[i, j].y * fdiff_y(phi, i, k)

    return a + b


@ti.func
def advect_kk_scheme(vc, phi, i, j):
    """Kawamura-Kuwabara Scheme

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#y2dbc484
    """
    coef = [-2, 10, -9, 2, -1]
    v = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])

    if vc[i, j].x < 0:
        v = ti.Vector(coef)
    else:
        v = -ti.Vector(coef[::-1])

    mx = ti.Matrix.cols(
        [
            sample(phi, i + 2, j),
            sample(phi, i + 1, j),
            sample(phi, i, j),
            sample(phi, i - 1, j),
            sample(phi, i - 2, j),
        ]
    )
    a = mx @ v / 6

    if vc[i, j].y < 0:
        v = ti.Vector(coef)
    else:
        v = -ti.Vector(coef[::-1])

    my = ti.Matrix.cols(
        [
            sample(phi, i, j + 2),
            sample(phi, i, j + 1),
            sample(phi, i, j),
            sample(phi, i, j - 1),
            sample(phi, i, j - 2),
        ]
    )
    b = my @ v / 6

    return vc[i, j].x * a + vc[i, j].y * b


@ti.func
def advect_eno(vc, phi, i, j):
    """ENO(Essentially Non-Oscillatory polynomial interpolation)

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#ufe6b856
    """
    # c2 = c3 = ti.Vector([0.0, 0.0])
    k = k_star = 0
    # calc advect_x
    # first order
    k = i if vc[i, j].x < 0.0 else i - 1

    Dk_1 = fdiff_x(phi, k, j)
    qx1 = Dk_1

    # second order
    Dk_2 = diff2_x(phi, k, j) / 2.0
    Dk1_2 = diff2_x(phi, k + 1, j) / 2.0

    c2 = Dk_2 if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x) else Dk1_2
    qx2 = c2 * (2 * (i - k) - 1)

    # third order
    k_star = k - 1 if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x) else k
    Dk_star_3 = diff3_x(phi, k_star, j) / 6.0
    Dk_star1_3 = diff3_x(phi, k_star + 1, j) / 6.0

    c3 = Dk_star_3 if ti.abs(Dk_star_3.x) <= ti.abs(Dk_star1_3.x) else Dk_star1_3
    qx3 = c3 * (3 * (i - k_star) ** 2 + 6 * (i - k_star) + 2)

    advect_x = vc[i, j].x * (qx1 + qx2 + qx3)

    # calc advect_y
    # first order
    k = j if vc[i, j].y < 0.0 else j - 1
    Dk_1 = fdiff_y(phi, i, k)
    qy1 = Dk_1

    # second order
    Dk_2 = diff2_y(phi, i, k) / 2.0
    Dk1_2 = diff2_y(phi, i, k + 1) / 2.0

    c2 = Dk_2 if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y) else Dk1_2
    qy2 = c2 * (2 * (j - k) - 1)

    # third order
    k_star = k - 1 if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y) else k
    Dk_star_3 = diff3_y(phi, i, k_star) / 6.0
    Dk_star1_3 = diff3_y(phi, i, k_star + 1) / 6.0

    c3 = Dk_star_3 if ti.abs(Dk_star_3.y) <= ti.abs(Dk_star1_3.y) else Dk_star1_3
    qy3 = c3 * (3 * (j - k_star) ** 2 + 6 * (j - k_star) + 2)

    advect_y = vc[i, j].y * (qy1 + qy2 + qy3)

    return advect_x + advect_y
