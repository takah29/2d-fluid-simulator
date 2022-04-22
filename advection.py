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
def advect(vc, i, j):
    """Central Differencing"""
    return vc[i, j].x * diff_x(vc, i, j) + vc[i, j].y * diff_y(vc, i, j)


@ti.func
def advect_upwind(vc, i, j):
    """Upwind differencing

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#tac8e468
    """
    a = b = ti.Vector([0.0, 0.0])
    if vc[i, j].x < 0.0:
        a = vc[i, j].x * fdiff_x(vc, i, j)
    else:
        a = vc[i, j].x * bdiff_x(vc, i, j)

    if vc[i, j].y < 0.0:
        b = vc[i, j].y * fdiff_y(vc, i, j)
    else:
        b = vc[i, j].y * bdiff_y(vc, i, j)

    return a + b


@ti.func
def advect_kk_scheme(vc, i, j):
    """Kawamura-Kuwabara Scheme

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#y2dbc484
    """
    a = b = ti.Vector([0.0, 0.0])
    if vc[i, j].x < 0:
        a = (
            vc[i, j].x
            * (
                -2 * sample(vc, i + 2, j)
                + 10 * sample(vc, i + 1, j)
                - 9 * sample(vc, i, j)
                + 2 * sample(vc, i - 1, j)
                - sample(vc, i - 2, j)
            )
            / 6
        )
    else:
        a = (
            vc[i, j].x
            * (
                sample(vc, i + 2, j)
                - 2 * sample(vc, i + 1, j)
                + 9 * sample(vc, i, j)
                - 10 * sample(vc, i - 1, j)
                + 2 * sample(vc, i - 2, j)
            )
            / 6
        )

    if vc[i, j].y < 0:
        b = (
            vc[i, j].y
            * (
                -2 * sample(vc, i, j + 2)
                + 10 * sample(vc, i, j + 1)
                - 9 * sample(vc, i, j)
                + 2 * sample(vc, i, j - 1)
                - sample(vc, i, j - 2)
            )
            / 6
        )
    else:
        b = (
            vc[i, j].y
            * (
                sample(vc, i, j + 2)
                - 2 * sample(vc, i, j + 1)
                + 9 * sample(vc, i, j)
                - 10 * sample(vc, i, j - 1)
                + 2 * sample(vc, i, j - 2)
            )
            / 6
        )

    return a + b


@ti.func
def advect_eno(vc, i, j):
    """ENO(Essentially Non-Oscillatory polynomial interpolation)

    http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#ufe6b856
    """
    c2 = c3 = ti.Vector([0.0, 0.0])
    k = k_star = 0
    # calc advect_x
    # first order
    if vc[i, j].x < 0.0:
        k = i
    else:
        k = i - 1

    Dk_1 = fdiff_x(vc, k, j)
    qx1 = Dk_1

    # second order
    Dk_2 = diff2_x(vc, k, j) / 2.0
    Dk1_2 = diff2_x(vc, k + 1, j) / 2.0

    if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x):
        c2 = Dk_2
    else:
        c2 = Dk1_2

    qx2 = c2 * (2 * (i - k) - 1)

    # third order
    if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x):
        k_star = k - 1
    else:
        k_star = k

    Dk_star_3 = diff3_x(vc, k_star, j) / 6.0
    Dk_star1_3 = diff3_x(vc, k_star + 1, j) / 6.0

    if ti.abs(Dk_star_3.x) <= ti.abs(Dk_star1_3.x):
        c3 = Dk_star_3
    else:
        c3 = Dk_star1_3

    qx3 = c3 * (3 * (i - k_star) ** 2 + 6 * (i - k_star) + 2)

    advect_x = vc[i, j].x * (qx1 + qx2 + qx3)

    # calc advect_y
    # first order
    if vc[i, j].y < 0.0:
        k = j
    else:
        k = j - 1

    Dk_1 = fdiff_y(vc, i, k)
    qy1 = Dk_1

    # second order
    Dk_2 = diff2_y(vc, i, k) / 2.0
    Dk1_2 = diff2_y(vc, i, k + 1) / 2.0

    if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y):
        c2 = Dk_2
    else:
        c2 = Dk1_2

    qy2 = c2 * (2 * (j - k) - 1)

    # third order
    if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y):
        k_star = k - 1
    else:
        k_star = k

    Dk_star_3 = diff3_y(vc, i, k_star) / 6.0
    Dk_star1_3 = diff3_y(vc, i, k_star + 1) / 6.0

    if ti.abs(Dk_star_3.y) <= ti.abs(Dk_star1_3.y):
        c3 = Dk_star_3
    else:
        c3 = Dk_star1_3

    qy3 = c3 * (3 * (j - k_star) ** 2 + 6 * (j - k_star) + 2)

    advect_y = vc[i, j].y * (qy1 + qy2 + qy3)

    return advect_x + advect_y
