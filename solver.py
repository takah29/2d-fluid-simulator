from abc import ABCMeta, abstractmethod

import taichi as ti

from differentiation import (
    sample,
    sign,
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
        return self.v.current, self.p.current, self.dyes.current

    @ti.kernel
    def _update_dyes(self, dn: ti.template(), dc: ti.template(), vc: ti.template()):
        for i, j in dn:
            if not self._bc.is_wall(i, j):
                dn[i, j] = ti.max(ti.min(dc[i, j] - self.dt * self._advect(vc, dc, i, j), 1.0), 0.0)


@ti.data_oriented
class CipMacSolver(Solver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, dt, Re, p_iter):
        super().__init__(boundary_condition)
        self.dt = dt
        self.Re = Re

        self.v = DoubleBuffers(self._resolution, 2)  # velocities
        self.p = DoubleBuffers(self._resolution, 1)  # pressure
        self.f = DoubleBuffers(self._resolution, 2)
        self.fx = DoubleBuffers(self._resolution, 2)
        self.fy = DoubleBuffers(self._resolution, 2)

        self.p_iter = p_iter

        # initial condition
        self._initialize()

    def _initialize(self):
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.f.reset()
        self.fx.reset()
        self.fy.reset()

        self._bc.calc(self.v.current, self.p.current)
        self.f.current.copy_from(self.v.current)
        self._calc_grad_x(self.fx.current, self.f.current)
        self._calc_grad_y(self.fy.current, self.f.current)

    def update(self):
        self._bc.calc(self.f.current, self.p.current)
        self._update_velocities(self.f, self.fx, self.fy, self.v, self.p)

        self._bc.calc(self.f.current, self.p.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

    def get_fields(self):
        return self.v.current, self.p.current

    @ti.kernel
    def _calc_grad_x(self, fx: ti.template(), f: ti.template()):
        for i, j in fx:
            if not self._bc.is_wall(i, j):
                fx[i, j] = diff_x(f, i, j)

    @ti.kernel
    def _calc_grad_y(self, fy: ti.template(), f: ti.template()):
        for i, j in fy:
            if not self._bc.is_wall(i, j):
                fy[i, j] = diff_y(f, i, j)

    def _update_velocities(self, f, fx, fy, v, p):
        self._non_advection_phase(f.next, f.current, p.current)
        self._non_advection_phase_grad(fx.next, fy.next, fx.current, fy.current, f.current, f.next)
        f.swap()
        fx.swap()
        fy.swap()
        self._advection_phase(
            f.next, fx.next, fy.next, f.current, fx.current, fy.current, v.current
        )
        f.swap()
        fx.swap()
        fy.swap()
        self.v.current.copy_from(self.f.current)

    @ti.kernel
    def _non_advection_phase(
        self,
        fn: ti.template(),
        fc: ti.template(),
        pc: ti.template(),
    ):
        """中間量の計算"""
        for i, j in fn:
            # 移流量の更新
            if not self._bc.is_wall(i, j):
                G = -ti.Vector(
                    [
                        diff_x(pc, i, j),
                        diff_y(pc, i, j),
                    ]
                ) + self._calc_diffusion(fc, i, j)
                fn[i, j] = fc[i, j] + G * self.dt

    @ti.kernel
    def _non_advection_phase_grad(
        self,
        fxn: ti.template(),
        fyn: ti.template(),
        fxc: ti.template(),
        fyc: ti.template(),
        fc: ti.template(),
        fn: ti.template(),
    ):
        """中間量の計算"""
        for i, j in fn:
            # 移流量の更新
            if not self._bc.is_wall(i, j):
                # 勾配の更新
                fxn[i, j] = (
                    fxc[i, j] + (fn[i + 1, j] - fc[i + 1, j] - fn[i - 1, j] + fc[i - 1, j]) / 2.0
                )
                fyn[i, j] = (
                    fyc[i, j] + (fn[i, j + 1] - fc[i, j + 1] - fn[i, j - 1] + fc[i, j - 1]) / 2.0
                )

    @ti.func
    def _cip_non_advect(self, fn, fxn, fyn, fc, fxc, fyc, pc, i, j):
        G = -ti.Vector(
            [
                diff_x(pc, i, j),
                diff_y(pc, i, j),
            ]
        ) + self._calc_diffusion(fc, i, j)
        fn[i, j] = fc[i, j] + G * self.dt

    @ti.func
    def _calc_diffusion(self, fc, i, j):
        return (diff2_x(fc, i, j) + diff2_y(fc, i, j)) / self.Re

    @ti.kernel
    def _advection_phase(
        self,
        fn: ti.template(),
        fxn: ti.template(),
        fyn: ti.template(),
        fc: ti.template(),
        fxc: ti.template(),
        fyc: ti.template(),
        vc: ti.template(),
    ):
        for i, j in fn:
            if not self._bc.is_wall(i, j):
                self._cip_advect(fn, fxn, fyn, fc, fxc, fyc, vc, i, j)

    @ti.func
    def _cip_advect(self, fn, fxn, fyn, fc, fxc, fyc, vc, i, j):
        i_s = int(sign(vc[i, j].x))
        j_s = int(sign(vc[i, j].y))
        i_m = i - i_s
        j_m = j - j_s

        tmp1 = fc[i, j] - fc[i, j_m] - fc[i_m, j] + fc[i_m, j_m]
        tmp2 = fc[i_m, j] - fc[i, j]
        tmp3 = fc[i, j_m] - fc[i, j]

        a = (i_s * (fxc[i_m, j] + fxc[i, j]) - 2.0 * (-tmp2)) / i_s
        b = (j_s * (fyc[i, j_m] + fyc[i, j]) - 2.0 * (-tmp3)) / j_s
        c = (-tmp1 - i_s * (fxc[i, j_m] - fxc[i, j])) / j_s
        d = (-tmp1 - j_s * (fyc[i_m, j] - fyc[i, j])) / i_s
        e = 3.0 * tmp2 + i_s * (fxc[i_m, j] + 2.0 * fxc[i, j])
        f = 3.0 * tmp3 + j_s * (fyc[i, j_m] + 2.0 * fyc[i, j])
        g = (-(fyc[i_m, j] - fyc[i, j]) + c) / i_s

        X = -vc[i, j].x * self.dt
        Y = -vc[i, j].y * self.dt

        fn[i, j] = (
            ((a * X + c * Y + e) * X + g * Y + fxc[i, j]) * X
            + ((b * Y + d * X + f) * Y + fyc[i, j]) * Y
            + fc[i, j]
        )

        # 勾配の更新
        Fx = (3.0 * a * X + 2.0 * c * Y + 2.0 * e) * X + (d * Y + g) * Y + fxc[i, j]
        Fy = (3.0 * b * Y + 2.0 * d * X + 2.0 * f) * Y + (c * X + g) * X + fyc[i, j]

        fxn[i, j] = Fx - self.dt * (Fx * diff_x(vc, i, j).x + Fy * diff_x(vc, i, j).y) / 2.0
        fyn[i, j] = Fy - self.dt * (Fx * diff_y(vc, i, j).x + Fy * diff_y(vc, i, j).y) / 2.0

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
class DyesCipMacSolver(CipMacSolver):
    """Maker And Cell method"""

    def __init__(self, boundary_condition, dt, Re, p_iter):
        self.dyes = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dyes
        self.dyesx = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dyes
        self.dyesy = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dyes

        super().__init__(boundary_condition, dt, Re, p_iter)

    def _initialize(self):
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.f.reset()
        self.fx.reset()
        self.fy.reset()
        self.dyes.reset()
        self.dyesx.reset()
        self.dyesy.reset()

        self._bc.calc(self.v.current, self.p.current, self.dyes.current)
        self.f.current.copy_from(self.v.current)
        self._calc_grad_x(self.fx.current, self.f.current)
        self._calc_grad_y(self.fy.current, self.f.current)

        self._calc_grad_x(self.dyesx.current, self.dyes.current)
        self._calc_grad_y(self.dyesy.current, self.dyes.current)

    def update(self):
        self._bc.calc(self.f.current, self.p.current, self.dyes.current)
        self._update_velocities(self.f, self.fx, self.fy, self.v, self.p)

        self._bc.calc(self.f.current, self.p.current, self.dyes.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.next, self.p.current, self.v.current)
            self.p.swap()

        self._bc.calc(self.v.current, self.p.current, self.dyes.current)
        self._update_dyes(
            self.dyes,
            self.dyesx,
            self.dyesy,
            self.v,
        )

    def get_fields(self):
        return self.v.current, self.p.current, self.dyes.current

    @ti.kernel
    def _non_advection_phase_dyes(
        self,
        dn: ti.template(),
        dc: ti.template(),
    ):
        """中間量の計算"""
        for i, j in dn:
            # 移流量の更新
            if not self._bc.is_wall(i, j):
                dn[i, j] = dc[i, j] + self._calc_diffusion(dc, i, j) * self.dt

    @ti.kernel
    def _clamp(self, d: ti.template(), dx: ti.template(), dy: ti.template()):
        for i, j in d:
            d[i, j] = ti.max(ti.min(d[i, j], 1.0), 0.0)
            dx[i, j] = ti.max(ti.min(dx[i, j], 1.0), -1.0)
            dy[i, j] = ti.max(ti.min(dy[i, j], 1.0), -1.0)

    def _update_dyes(self, dyes, dyesx, dyesy, v):
        self._non_advection_phase_dyes(dyes.next, dyes.current)
        self._non_advection_phase_grad(
            dyesx.next, dyesy.next, dyesx.current, dyesy.current, dyes.current, dyes.next
        )
        dyes.swap()
        dyesx.swap()
        dyesy.swap()

        self._advection_phase(
            dyes.next, dyesx.next, dyesy.next, dyes.current, dyesx.current, dyesy.current, v.current
        )
        dyes.swap()
        dyesx.swap()
        dyesy.swap()
        self._clamp(dyes.current, dyesx.current, dyesy.current)
