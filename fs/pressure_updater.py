from abc import ABCMeta, abstractmethod

import taichi as ti

from fs.boundary_condition import BoundaryCondition, DyeBoundaryCondition
from fs.differentiation import sample
from fs.double_buffer import DoubleBuffer


class PressureUpdater(metaclass=ABCMeta):
    def __init__(
        self, boundary_condition: BoundaryCondition | DyeBoundaryCondition, dt: float, dx: float
    ) -> None:
        self._bc = boundary_condition
        self.dt = dt
        self.dx = dx

    @abstractmethod
    def update(self, p: DoubleBuffer, v_current: ti.Field) -> None:
        pass


@ti.func
def predict_p(pc: ti.template(), vc: ti.template(), i: int, j: int, dt: float, dx: float):  # type: ignore[valid-type]  # noqa: ANN201
    sub_x = sample(vc, i + 1, j) - sample(vc, i - 1, j)
    sub_y = sample(vc, i, j + 1) - sample(vc, i, j - 1)

    return (
        0.25
        * (
            sample(pc, i + 1, j)
            + sample(pc, i - 1, j)
            + sample(pc, i, j + 1)
            + sample(pc, i, j - 1)
        )
        + (sub_x.x**2 + sub_y.y**2 + (sub_y.x * sub_x.y)) / 8.0  # type: ignore[reportAttributeAccessIssue]
        - dx * (sub_x.x + sub_y.y) / (8 * dt)  # type: ignore[reportAttributeAccessIssue]
    )


@ti.data_oriented
class JacobiPressureUpdater(PressureUpdater):
    """Jacobi Method."""

    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        dt: float,
        dx: float,
        n_iter: int,
    ) -> None:
        super().__init__(boundary_condition, dt, dx)

        self._n_iter = n_iter

    def update(self, p: DoubleBuffer, v_current: ti.Field) -> None:
        for _ in range(self._n_iter):
            self._bc.set_pressure_boundary_condition(p.current)
            self._update(p.next, p.current, v_current)
            p.swap()

    @ti.kernel
    def _update(self, p_next: ti.template(), p_current: ti.template(), v_current: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in p_next:
            if not self._bc.is_wall(i, j):
                p_next[i, j] = predict_p(p_current, v_current, i, j, self.dt, self.dx)


@ti.data_oriented
class RedBlackSorPressureUpdater(PressureUpdater):
    """Red-Black SOR Method."""

    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        dt: float,
        dx: float,
        relaxation_factor: float,
        n_iter: int,
    ) -> None:
        super().__init__(boundary_condition, dt, dx)

        self._n_iter = n_iter
        self._relaxation_factor = relaxation_factor

    def update(self, p: DoubleBuffer, v_current: ti.Field) -> None:
        for _ in range(self._n_iter):
            self._bc.set_pressure_boundary_condition(p.current)
            self._update(p.next, p.current, v_current)
            p.swap()

    def _update(self, p_next: ti.Field, p_current: ti.Field, v_current: ti.Field) -> None:
        # 圧力のFieldは1つでも良いがインターフェイスの統一のために2つ受け取るようにしている
        # Fieldを1つしか使わない場合はpn, pcを同じFieldとして与えれば良い
        self._update_pressures_odd(p_next, p_current, v_current)
        self._update_pressures_even(p_next, p_next, v_current)

    @ti.kernel
    def _update_pressures_odd(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in pn:
            if (i + j) % 2 == 1 and self._bc.is_fluid_domain(i, j):
                pn[i, j] = self._pn_ij(pc, vc, i, j)

    @ti.kernel
    def _update_pressures_even(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in pn:
            if (i + j) % 2 == 0 and self._bc.is_fluid_domain(i, j):
                pn[i, j] = self._pn_ij(pc, vc, i, j)

    @ti.func
    def _pn_ij(self, pc: ti.template(), vc: ti.template(), i: int, j: int):  # type: ignore[valid-type]  # noqa: ANN202
        return (1.0 - self._relaxation_factor) * pc[i, j] + self._relaxation_factor * predict_p(
            pc, vc, i, j, self.dt, self.dx
        )
