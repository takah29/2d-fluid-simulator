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
def fdiff_x(field, i, j, dx):
    """Forward Difference x"""
    return (sample(field, i + 1, j) - sample(field, i, j)) / dx


@ti.func
def fdiff_y(field, i, j, dx):
    """Forward Difference y"""
    return (sample(field, i, j + 1) - sample(field, i, j)) / dx


@ti.func
def bdiff_x(field, i, j, dx):
    """Backward Difference x"""
    return (sample(field, i, j) - sample(field, i - 1, j)) / dx


@ti.func
def bdiff_y(field, i, j, dx):
    """Backward Difference y"""
    return (sample(field, i, j) - sample(field, i, j - 1)) / dx


@ti.func
def diff_x(field, i, j, dx):
    """Central Difference x"""
    return 0.5 * (sample(field, i + 1, j) - sample(field, i - 1, j)) / dx


@ti.func
def diff_y(field, i, j, dx):
    """Central Difference y"""
    return 0.5 * (sample(field, i, j + 1) - sample(field, i, j - 1)) / dx


@ti.func
def diff2_x(field, i, j, dx):
    return (sample(field, i + 1, j) - 2.0 * sample(field, i, j) + sample(field, i - 1, j)) / dx**2


@ti.func
def diff2_y(field, i, j, dx):
    return (sample(field, i, j + 1) - 2.0 * sample(field, i, j) + sample(field, i, j - 1)) / dx**2
