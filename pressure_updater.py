from abc import ABCMeta, abstractmethod

import taichi as ti

from differentiation import sample, diff_x, diff_y


class PressureUpdater(metaclass=ABCMeta):
    def __init__(self, boundary_condition, dt):
        self._bc = boundary_condition
        self.dt = dt

    @abstractmethod
    def update(self, p, v_current):
        pass


@ti.func
def predict_p(pc, vc, dt, i, j):
    dx = diff_x(vc, i, j)
    dy = diff_y(vc, i, j)
    pred_p = (
        (sample(pc, i + 1, j) + sample(pc, i - 1, j) + sample(pc, i, j + 1) + sample(pc, i, j - 1))
        - (dx.x + dy.y) / dt
        + dx.x**2
        + dy.y**2
        + 2 * dy.x * dx.y
    ) * 0.25

    return pred_p


@ti.data_oriented
class JacobiPressureUpdater(PressureUpdater):
    """Jacobi Method"""

    def __init__(self, boundary_condition, dt, n_iter):
        super().__init__(boundary_condition, dt)

        self._n_iter = n_iter

    def update(self, p, v_current):
        for _ in range(self._n_iter):
            self._bc.set_pressure_boundary_condition(p.current)
            self._update(p.next, p.current, v_current)
            p.swap()

    @ti.kernel
    def _update(self, p_next: ti.template(), p_current: ti.template(), v_current: ti.template()):
        for i, j in p_next:
            if not self._bc.is_wall(i, j):
                p_next[i, j] = predict_p(p_current, v_current, self.dt, i, j)


@ti.data_oriented
class RedBlackSorPressureUpdater(PressureUpdater):
    """Red-Black SOR Method"""

    def __init__(self, boundary_condition, dt, relaxation_factor, n_iter):
        super().__init__(boundary_condition, dt)

        self._n_iter = n_iter
        self._relaxation_factor = relaxation_factor

    def update(self, p, v_current):
        for _ in range(self._n_iter):
            self._bc.set_pressure_boundary_condition(p.current)
            self._update(p.next, p.current, v_current)
            p.swap()

    def _update(self, p_next, p_current, v_current):
        # 圧力のFieldは1つでも良いがインターフェイスの統一のために2つ受け取るようにしている
        # Fieldを1つしか使わない場合はpn, pcを同じFieldとして与えれば良い
        self._update_pressures_odd(p_next, p_current, v_current)
        self._update_pressures_even(p_next, p_next, v_current)

    @ti.kernel
    def _update_pressures_odd(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):
        for i, j in pn:
            if (i + j) % 2 == 1:
                if self._bc.is_fluid_domain(i, j):
                    pn[i, j] = self._pn_ij(pc, vc, i, j)

    @ti.kernel
    def _update_pressures_even(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):
        for i, j in pn:
            if (i + j) % 2 == 0:
                if self._bc.is_fluid_domain(i, j):
                    pn[i, j] = self._pn_ij(pc, vc, i, j)

    @ti.func
    def _pn_ij(self, pc, vc, i, j):
        return (1.0 - self._relaxation_factor) * pc[i, j] + self._relaxation_factor * predict_p(
            pc, vc, self.dt, i, j
        )
