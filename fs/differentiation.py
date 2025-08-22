import taichi as ti


@ti.func
def sample(field: ti.template(), i: int, j: int) -> ti.Vector | float:  # type: ignore[valid-type]
    i = ti.max(0, ti.min(field.shape[0] - 1, i))
    j = ti.max(0, ti.min(field.shape[1] - 1, j))
    idx = ti.Vector([int(i), int(j)])
    return field[idx]


@ti.func
def sign(x: float) -> float:
    return -1.0 if x < 0.0 else 1.0


@ti.func
def fdiff_x(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Forward Difference x."""
    return (sample(field, i + 1, j) - sample(field, i, j)) / dx


@ti.func
def fdiff_y(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Forward Difference y."""
    return (sample(field, i, j + 1) - sample(field, i, j)) / dx


@ti.func
def bdiff_x(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Backward Difference x."""
    return (sample(field, i, j) - sample(field, i - 1, j)) / dx


@ti.func
def bdiff_y(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Backward Difference y."""
    return (sample(field, i, j) - sample(field, i, j - 1)) / dx


@ti.func
def diff_x(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Central Difference x."""
    return 0.5 * (sample(field, i + 1, j) - sample(field, i - 1, j)) / dx


@ti.func
def diff_y(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    """Central Difference y."""
    return 0.5 * (sample(field, i, j + 1) - sample(field, i, j - 1)) / dx


@ti.func
def diff2_x(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    return (sample(field, i + 1, j) - 2.0 * sample(field, i, j) + sample(field, i - 1, j)) / dx**2


@ti.func
def diff2_y(field: ti.template(), i: int, j: int, dx: float) -> ti.Vector | float:  # type: ignore[valid-type]
    return (sample(field, i, j + 1) - 2.0 * sample(field, i, j) + sample(field, i, j - 1)) / dx**2
