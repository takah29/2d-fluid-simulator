import taichi as ti


@ti.func
def sample(field, i, j):
    i = ti.max(0, ti.min(field.shape[0] - 1, i))
    j = ti.max(0, ti.min(field.shape[1] - 1, j))
    idx = ti.Vector([int(i), int(j)])
    return field[idx]


@ti.func
def sign(x):
    return -1.0 if x < 0.0 else 1.0


@ti.func
def fdiff_x(field, i, j):
    """Forward Difference x"""
    return sample(field, i + 1, j) - sample(field, i, j)


@ti.func
def fdiff_y(field, i, j):
    """Forward Difference y"""
    return sample(field, i, j + 1) - sample(field, i, j)


@ti.func
def bdiff_x(field, i, j):
    """Backward Difference x"""
    return sample(field, i, j) - sample(field, i - 1, j)


@ti.func
def bdiff_y(field, i, j):
    """Backward Difference y"""
    return sample(field, i, j) - sample(field, i, j - 1)


@ti.func
def diff_x(field, i, j):
    """Central Difference x"""
    return 0.5 * (sample(field, i + 1, j) - sample(field, i - 1, j))


@ti.func
def diff_y(field, i, j):
    """Central Difference y"""
    return 0.5 * (sample(field, i, j + 1) - sample(field, i, j - 1))


@ti.func
def diff2_x(field, i, j):
    return sample(field, i + 1, j) - 2.0 * sample(field, i, j) + sample(field, i - 1, j)


@ti.func
def diff2_y(field, i, j):
    return sample(field, i, j + 1) - 2.0 * sample(field, i, j) + sample(field, i, j - 1)


@ti.func
def diff3_x(field, i, j):
    return (
        sample(field, i + 2, j)
        - 3.0 * sample(field, i + 1, j)
        + 3.0 * sample(field, i, j)
        - sample(field, i - 1, j)
    )


@ti.func
def diff3_y(field, i, j):
    return (
        sample(field, i, j + 2)
        - 3.0 * sample(field, i, j + 1)
        + 3.0 * sample(field, i, j)
        - sample(field, i, j - 1)
    )
