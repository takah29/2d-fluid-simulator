from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import taichi as ti

from boundary_condition import BoundaryCondition, DyeBoundaryCondition
from differentiation import diff2_x, diff2_y, diff_x, diff_y, sign
from pressure_updater import PressureUpdater
from vorticity_confinement import VorticityConfinement

VELOCITY_LIMIT = 10.0


class DoubleBuffers:
    def __init__(self, resolution, n_channel: int) -> None:
        if n_channel == 1:
            self.current = ti.field(ti.f32, shape=resolution)
            self.next = ti.field(ti.f32, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, ti.f32, shape=resolution)
            self.next = ti.Vector.field(n_channel, ti.f32, shape=resolution)

    def swap(self) -> None:
        self.current, self.next = self.next, self.current

    def reset(self) -> None:
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class Solver(metaclass=ABCMeta):
    def __init__(self, boundary_condition: BoundaryCondition | DyeBoundaryCondition) -> None:
        self._bc = boundary_condition
        self.resolution = boundary_condition.get_resolution()

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def get_fields(self):  # noqa: ANN201
        pass

    @ti.func
    def is_wall(self, i: int, j: int) -> bool:
        return self._bc.is_wall(i, j)

    @ti.func
    def is_fluid_domain(self, i: int, j: int) -> bool:
        return self._bc.is_fluid_domain(i, j)


@ti.kernel
def limit_field(field: ti.template(), limit: ti.f32):  # type: ignore[valid-type]  # noqa: ANN201
    for i, j in field:
        norm = field[i, j].norm()
        if norm > limit:
            field[i, j] = limit * (field[i, j] / norm)


@ti.kernel
def clamp_field(field: ti.template(), low: ti.f32, high: ti.f32):  # type: ignore[valid-type]  # noqa: ANN201
    for i, j in field:
        field[i, j] = ti.min(ti.max(field[i, j], low), high)


@ti.data_oriented
class MacSolver(Solver):
    """Maker And Cell method."""

    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        pressure_updater: PressureUpdater,
        advect_function: Callable[[ti.Field, ti.Field, int, int, float], float],
        dt: float,
        dx: float,
        re: float,
        vorticity_confinement: VorticityConfinement | None = None,
    ) -> None:
        super().__init__(boundary_condition)

        self._advect = advect_function
        self.dt = dt
        self.dx = dx
        self.re = re

        self.pressure_updater = pressure_updater
        self.vorticity_confinement = vorticity_confinement

        self.v = DoubleBuffers(self.resolution, 2)  # velocity
        self.p = DoubleBuffers(self.resolution, 1)  # pressure

    def update(self) -> None:
        self._bc.set_velocity_boundary_condition(self.v.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self.v.swap()

        if self.vorticity_confinement is not None:
            self.vorticity_confinement.apply(self.v)
            self.v.swap()
        self.pressure_updater.update(self.p, self.v.current)

        limit_field(self.v.current, VELOCITY_LIMIT)

    def get_fields(self) -> tuple[ti.Field, ti.Field]:  # type: ignore[override]
        return self.v.current, self.p.current

    @ti.kernel
    def _update_velocities(self, vn: ti.template(), vc: ti.template(), pc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in vn:
            if self.is_fluid_domain(i, j):
                vn[i, j] = vc[i, j] + self.dt * (
                    -self._advect(vc, vc, i, j, self.dx)
                    - ti.Vector(
                        [
                            diff_x(pc, i, j, self.dx),
                            diff_y(pc, i, j, self.dx),
                        ]
                    )
                    + (diff2_x(vc, i, j, self.dx) + diff2_y(vc, i, j, self.dx)) / self.re
                )


@ti.data_oriented
class DyeMacSolver(MacSolver):
    """Maker And Cell method."""

    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        pressure_updater: PressureUpdater,
        advect_function: Callable[[ti.Field, ti.Field, int, int, float], float],
        dt: float,
        dx: float,
        re: float,
        vorticity_confinement: VorticityConfinement | None = None,
    ) -> None:
        super().__init__(
            boundary_condition,
            pressure_updater,
            advect_function,
            dt,
            dx,
            re,
            vorticity_confinement,
        )

        self.dye = DoubleBuffers(self.resolution, 3)

    def update(self) -> None:
        self._bc.set_velocity_boundary_condition(self.v.current)
        self._update_velocities(self.v.next, self.v.current, self.p.current)
        self.v.swap()

        if self.vorticity_confinement is not None:
            self.vorticity_confinement.apply(self.v)
            self.v.swap()

        self.pressure_updater.update(self.p, self.v.current)

        limit_field(self.v.current, VELOCITY_LIMIT)

        self._bc.set_dye_boundary_condition(self.dye.current)  # type: ignore[union-attr]
        self._update_dye(self.dye.next, self.dye.current, self.v.current)
        self.dye.swap()
        clamp_field(self.dye.current, 0.0, 1.0)

    def get_fields(self) -> tuple[ti.Field, ti.Field, ti.Field]:  # type: ignore[override]
        return self.v.current, self.p.current, self.dye.current

    @ti.kernel
    def _update_dye(self, dn: ti.template(), dc: ti.template(), vc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in dn:
            if self.is_fluid_domain(i, j):
                dn[i, j] = dc[i, j] - self.dt * self._advect(vc, dc, i, j, self.dx)


@ti.data_oriented
class CipMacSolver(Solver):
    """Maker And Cell method."""

    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        pressure_updater: PressureUpdater,
        dt: float,
        dx: float,
        re: float,
        vorticity_confinement: VorticityConfinement | None = None,
    ) -> None:
        super().__init__(boundary_condition)
        self.dt = dt
        self.dx = dx
        self.re = re

        self.pressure_updater = pressure_updater
        self.vorticity_confinement = vorticity_confinement

        self.v = DoubleBuffers(self.resolution, 2)  # velocity
        self.vx = DoubleBuffers(self.resolution, 2)  # velocity gradient x
        self.vy = DoubleBuffers(self.resolution, 2)  # velocity gradient y
        self.p = DoubleBuffers(self.resolution, 1)  # pressure

        self._set_grad(self.vx.current, self.vy.current, self.v.current)

    def update(self) -> None:
        self._bc.set_velocity_boundary_condition(self.v.current)
        self._update_velocities(self.v, self.vx, self.vy, self.p)

        if self.vorticity_confinement is not None:
            self.vorticity_confinement.apply(self.v)
            self.v.swap()

        self.pressure_updater.update(self.p, self.v.current)

        limit_field(self.v.current, VELOCITY_LIMIT)

    def get_fields(self) -> tuple[ti.Field, ti.Field]:  # type: ignore[override]
        return self.v.current, self.p.current

    @ti.kernel
    def _set_grad(self, fx: ti.template(), fy: ti.template(), f: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in fx:
            fx[i, j] = diff_x(f, i, j, self.dx)
            fy[i, j] = diff_y(f, i, j, self.dx)

    def _update_velocities(
        self, v: DoubleBuffers, vx: DoubleBuffers, vy: DoubleBuffers, p: DoubleBuffers
    ) -> None:
        self._non_advection_phase(v.next, v.current, p.current)
        self._non_advection_phase_grad(vx.next, vy.next, vx.current, vy.current, v.current, v.next)
        v.swap()
        vx.swap()
        vy.swap()

        self._advection_phase(
            v.next, vx.next, vy.next, v.current, vx.current, vy.current, v.current
        )
        v.swap()
        vx.swap()
        vy.swap()

    @ti.kernel
    def _non_advection_phase(self, fn: ti.template(), fc: ti.template(), pc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        """中間量の計算."""
        for i, j in fn:
            if not self.is_wall(i, j):
                g = -ti.Vector(
                    [
                        diff_x(pc, i, j, self.dx),
                        diff_y(pc, i, j, self.dx),
                    ]
                ) + self._calc_diffusion(fc, i, j)
                fn[i, j] = fc[i, j] + g * self.dt

    @ti.kernel
    def _non_advection_phase_grad(  # noqa: ANN202
        self,
        fxn: ti.template(),  # type: ignore[valid-type]
        fyn: ti.template(),  # type: ignore[valid-type]
        fxc: ti.template(),  # type: ignore[valid-type]
        fyc: ti.template(),  # type: ignore[valid-type]
        fc: ti.template(),  # type: ignore[valid-type]
        fn: ti.template(),  # type: ignore[valid-type]
    ):
        """中間量の勾配の計算."""
        for i, j in fn:
            if not self.is_wall(i, j):
                # 勾配の更新
                fxn[i, j] = fxc[i, j] + (
                    fn[i + 1, j] - fc[i + 1, j] - fn[i - 1, j] + fc[i - 1, j]
                ) / (2.0 * self.dx)
                fyn[i, j] = fyc[i, j] + (
                    fn[i, j + 1] - fc[i, j + 1] - fn[i, j - 1] + fc[i, j - 1]
                ) / (2.0 * self.dx)

    @ti.func
    def _calc_diffusion(self, fc: ti.template(), i: int, j: int) -> float:  # type: ignore[valid-type]
        return (diff2_x(fc, i, j, self.dx) + diff2_y(fc, i, j, self.dx)) / self.re

    @ti.kernel
    def _advection_phase(  # noqa: ANN202
        self,
        fn: ti.template(),  # type: ignore[valid-type]
        fxn: ti.template(),  # type: ignore[valid-type]
        fyn: ti.template(),  # type: ignore[valid-type]
        fc: ti.template(),  # type: ignore[valid-type]
        fxc: ti.template(),  # type: ignore[valid-type]
        fyc: ti.template(),  # type: ignore[valid-type]
        v: ti.template(),  # type: ignore[valid-type]
    ):
        for i, j in fn:
            if self.is_fluid_domain(i, j):
                self._cip_advect(fn, fxn, fyn, fc, fxc, fyc, v, i, j)

    @ti.func
    def _cip_advect(
        self,
        fn: ti.template(),  # type: ignore[valid-type]
        fxn: ti.template(),  # type: ignore[valid-type]
        fyn: ti.template(),  # type: ignore[valid-type]
        fc: ti.template(),  # type: ignore[valid-type]
        fxc: ti.template(),  # type: ignore[valid-type]
        fyc: ti.template(),  # type: ignore[valid-type]
        v: ti.template(),  # type: ignore[valid-type]
        i: int,
        j: int,
    ) -> None:
        i_s = int(sign(v[i, j].x))
        j_s = int(sign(v[i, j].y))
        i_m = i - i_s
        j_m = j - j_s

        tmp1 = fc[i, j] - fc[i, j_m] - fc[i_m, j] + fc[i_m, j_m]
        tmp2 = fc[i_m, j] - fc[i, j]
        tmp3 = fc[i, j_m] - fc[i, j]

        i_s_denom = i_s * self.dx**3
        j_s_denom = j_s * self.dx**3

        a = (i_s * (fxc[i_m, j] + fxc[i, j]) * self.dx - 2.0 * (-tmp2)) / i_s_denom
        b = (j_s * (fyc[i, j_m] + fyc[i, j]) * self.dx - 2.0 * (-tmp3)) / j_s_denom
        c = (-tmp1 - i_s * (fxc[i, j_m] - fxc[i, j]) * self.dx) / j_s_denom
        d = (-tmp1 - j_s * (fyc[i_m, j] - fyc[i, j]) * self.dx) / i_s_denom
        e = (3.0 * tmp2 + i_s * (fxc[i_m, j] + 2.0 * fxc[i, j]) * self.dx) / self.dx**2
        f = (3.0 * tmp3 + j_s * (fyc[i, j_m] + 2.0 * fyc[i, j]) * self.dx) / self.dx**2
        g = (-(fyc[i_m, j] - fyc[i, j]) + c * self.dx**2) / (i_s * self.dx)

        X = -v[i, j].x * self.dt  # noqa: N806
        Y = -v[i, j].y * self.dt  # noqa: N806

        # 移流量の更新
        fn[i, j] = (
            ((a * X + c * Y + e) * X + g * Y + fxc[i, j]) * X
            + ((b * Y + d * X + f) * Y + fyc[i, j]) * Y
            + fc[i, j]
        )

        # 勾配の更新
        Fx = (3.0 * a * X + 2.0 * c * Y + 2.0 * e) * X + (d * Y + g) * Y + fxc[i, j]  # noqa: N806
        Fy = (3.0 * b * Y + 2.0 * d * X + 2.0 * f) * Y + (c * X + g) * X + fyc[i, j]  # noqa: N806

        dx = diff_x(v, i, j, self.dx)
        dy = diff_y(v, i, j, self.dx)
        fxn[i, j] = Fx - self.dt * (Fx * dx.x + Fy * dx.y) / 2.0  # type: ignore[reportAttributeAccessIssue]
        fyn[i, j] = Fy - self.dt * (Fx * dy.x + Fy * dy.y) / 2.0  # type: ignore[reportAttributeAccessIssue]


@ti.data_oriented
class DyeCipMacSolver(CipMacSolver):
    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        pressure_updater: PressureUpdater,
        dt: float,
        dx: float,
        re: float,
        vorticity_confinement: VorticityConfinement | None = None,
    ) -> None:
        super().__init__(boundary_condition, pressure_updater, dt, dx, re, vorticity_confinement)

        self.dye = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye
        self.dyex = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye gradient x
        self.dyey = DoubleBuffers(boundary_condition.get_resolution(), 3)  # dye gradient y
        self._set_grad(self.dyex.current, self.dyey.current, self.dye.current)

    def update(self) -> None:
        self._bc.set_velocity_boundary_condition(self.v.current)
        self._update_velocities(self.v, self.vx, self.vy, self.p)

        if self.vorticity_confinement is not None:
            self.vorticity_confinement.apply(self.v)
            self.v.swap()

        self.pressure_updater.update(self.p, self.v.current)

        # 発散しないように流速を制限する。精度が低下する。
        limit_field(self.v.current, VELOCITY_LIMIT)

        self._bc.set_dye_boundary_condition(self.dye.current)  # type: ignore[union-attr]
        self._update_dye(
            self.dye,
            self.dyex,
            self.dyey,
            self.v,
        )
        clamp_field(self.dye.current, 0.0, 1.0)

    def get_fields(self) -> tuple[ti.Field, ti.Field, ti.Field]:  # type: ignore[override]
        return self.v.current, self.p.current, self.dye.current

    @ti.kernel
    def _non_advection_phase_dye(self, dn: ti.template(), dc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        """中間量の計算."""
        for i, j in dn:
            if not self.is_wall(i, j):
                dn[i, j] = dc[i, j] + self._calc_diffusion(dc, i, j) * self.dt

    def _update_dye(
        self, dye: DoubleBuffers, dyex: DoubleBuffers, dyey: DoubleBuffers, v: DoubleBuffers
    ) -> None:
        self._non_advection_phase_dye(dye.next, dye.current)
        self._non_advection_phase_grad(
            dyex.next, dyey.next, dyex.current, dyey.current, dye.current, dye.next
        )
        dye.swap()
        dyex.swap()
        dyey.swap()

        self._advection_phase(
            dye.next, dyex.next, dyey.next, dye.current, dyex.current, dyey.current, v.current
        )
        dye.swap()
        dyex.swap()
        dyey.swap()
