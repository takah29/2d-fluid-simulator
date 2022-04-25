from abc import ABCMeta, abstractmethod

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
)


class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(float, shape=resolution)
            self.next = ti.field(float, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, float, shape=resolution)
            self.next = ti.Vector.field(n_channel, float, shape=resolution)

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class Solver(metaclass=ABCMeta):
    def __init__(self, boundary_condition):
        self._bc = boundary_condition
        self._resolution = boundary_condition.get_resolution()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def get_fields(self):
        pass

    @ti.func
    def is_wall(self, i, j):
        return self._bc.is_wall(i, j)


@ti.data_oriented
class MacSolver(Solver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, advect_function, dt, Re, p_iter):
        super().__init__(boundary_condition)

        self._advect = advect_function

        self.dt = dt
        self.Re = Re

        self.v = DoubleBuffers(self._resolution, 2)  # velocities
        self.p = DoubleBuffers(self._resolution, 1)  # pressure

        self.p_iter = p_iter

        # initial condition
        self.v.current.fill(ti.Vector([0.4, 0.0]))

    def update(self):
        self._bc.calc(self.v.current, self.p.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self.v.swap()

        self._bc.calc(self.v.current, self.p.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

    def get_fields(self):
        return self.v.current, self.p.current

    @ti.kernel
    def _update_velocities(self, vn: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in vn:
            if not self._bc.is_wall(i, j):
                vn[i, j] = vc[i, j] + self.dt * (
                    -self._advect(vc, vc, i, j)
                    - ti.Vector(
                        [
                            diff_x(pc, i, j),
                            diff_y(pc, i, j),
                        ]
                    )
                    + (diff2_x(vc, i, j) + diff2_y(vc, i, j)) / self.Re
                )

    @ti.kernel
    def _update_pressures(self, pn: ti.template(), pc: ti.template(), vc: ti.template()):
        for i, j in pn:
            if not self._bc.is_wall(i, j):
                pn[i, j] = (
                    (
                        sample(pc, i + 1, j)
                        + sample(pc, i - 1, j)
                        + sample(pc, i, j + 1)
                        + sample(pc, i, j - 1)
                    )
                    - (diff_x(vc, i, j).x + diff_y(vc, i, j).y) / self.dt
                    + diff_x(vc, i, j).x ** 2
                    + diff_y(vc, i, j).y ** 2
                    + 2 * diff_y(vc, i, j).x * diff_x(vc, i, j).y
                ) * 0.25


@ti.data_oriented
class FsSolver(Solver):
    """Fractional Step method"""

    def __init__(self, boundary_condition, advect_function, dt, Re, p_iter):
        super().__init__(boundary_condition)

        self._advect = advect_function

        self.dt = dt
        self.Re = Re

        self.v = ti.Vector.field(2, float, shape=self._resolution)  # velocities
        self.p = DoubleBuffers(self._resolution, 1)  # pressure
        self.tv = ti.Vector.field(2, float, shape=self._resolution)  # temp velocities

        self.p_iter = p_iter

        # initial condition
        self.v.fill(ti.Vector([0.4, 0.0]))

    def update(self):
        self._bc.calc(self.v, self.p.current)
        self._calc_temp_velocities(self.tv, self.v)

        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.tv)
            self.p.swap()

        self._update_velocities(self.v, self.tv, self.p.current)

    def get_fields(self):
        return self.v, self.p.current

    @ti.kernel
    def _calc_temp_velocities(self, tv: ti.template(), v: ti.template()):
        for i, j in tv:
            if not self._bc.is_wall(i, j):
                tv[i, j] = v[i, j] + self.dt * (
                    -self._advect(v, v, i, j) + (diff2_x(v, i, j) + diff2_y(v, i, j)) / self.Re
                )

    @ti.kernel
    def _update_pressures(self, pn: ti.template(), pc: ti.template(), tv: ti.template()):
        for i, j in pn:
            if not self._bc.is_wall(i, j):
                pn[i, j] = 0.25 * (
                    (
                        sample(pc, i + 1, j)
                        + sample(pc, i - 1, j)
                        + sample(pc, i, j + 1)
                        + sample(pc, i, j - 1)
                    )
                    - (diff_x(tv, i, j).x + diff_y(tv, i, j).y) / self.dt
                )

    @ti.kernel
    def _update_velocities(self, v: ti.template(), tv: ti.template(), pc: ti.template()):
        for i, j in v:
            if not self._bc.is_wall(i, j):
                v[i, j] = tv[i, j] - self.dt * ti.Vector([diff_x(pc, i, j), diff_y(pc, i, j)])


@ti.data_oriented
class DyesMacSolver(MacSolver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, advect_function, dt, Re, p_iter):
        super().__init__(boundary_condition, advect_function, dt, Re, p_iter)
        self.dyes = DoubleBuffers(self._resolution, 3)  # dyes

    def update(self):
        self._bc.calc(self.v.current, self.p.current, self.dyes.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self.v.swap()

        self._bc.calc(self.v.current, self.p.current, self.dyes.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

        self._bc.calc(self.v.current, self.p.current, self.dyes.current)
        self._update_dyes(self.dyes.next, self.dyes.current, self.v.current)
        self.dyes.swap()

    def get_fields(self):
        return self.dyes.current, self.v.current, self.p.current

    @ti.kernel
    def _update_dyes(self, dn: ti.template(), dc: ti.template(), vc: ti.template()):
        for i, j in dn:
            if not self._bc.is_wall(i, j):
                dn[i, j] = dc[i, j] - self.dt * self._advect(vc, dc, i, j)