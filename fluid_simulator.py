import taichi as ti

from advection import advect_kk_scheme, advect_upwind
from boundary_condition import (
    create_boundary_condition1,
    create_boundary_condition2,
    create_boundary_condition3,
    create_boundary_condition4,
    create_boundary_condition5,
    create_boundary_condition6,
)
from solver import CipMacSolver, DyeCipMacSolver, DyeMacSolver, MacSolver
from pressure_updater import RedBlackSorPressureUpdater
from vorticity_confinement import VorticityConfinement
from visualization import visualize_norm, visualize_pressure, visualize_vorticity


def get_boundary_condition(num, resolution, no_dye):
    if num == 1:
        boundary_condition = create_boundary_condition1(resolution, no_dye)
    elif num == 2:
        boundary_condition = create_boundary_condition2(resolution, no_dye)
    elif num == 3:
        boundary_condition = create_boundary_condition3(resolution, no_dye)
    elif num == 4:
        boundary_condition = create_boundary_condition4(resolution, no_dye)
    elif num == 5:
        boundary_condition = create_boundary_condition5(resolution, no_dye)
    elif num == 6:
        boundary_condition = create_boundary_condition6(resolution, no_dye)
    else:
        raise NotImplementedError

    return boundary_condition


@ti.data_oriented
class FluidSimulator:
    def __init__(self, solver):
        self._solver = solver
        self.rgb_buf = ti.Vector.field(3, ti.f32, shape=solver._resolution)  # image buffer
        self._wall_color = ti.Vector([0.5, 0.7, 0.5])

    def step(self):
        self._solver.update()

    def get_norm_field(self):
        self._to_norm(self.rgb_buf, *self._solver.get_fields()[:2])
        return self.rgb_buf

    def get_pressure_field(self):
        self._to_pressure(self.rgb_buf, self._solver.get_fields()[1])
        return self.rgb_buf

    def get_vorticity_field(self):
        self._to_vorticity(self.rgb_buf, self._solver.get_fields()[0])
        return self.rgb_buf

    @ti.kernel
    def _to_norm(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.015 * visualize_norm(vc[i, j])
            rgb_buf[i, j] += 0.00002 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_pressure(self, rgb_buf: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.0002 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_vorticity(self, rgb_buf: ti.template(), vc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.2 * visualize_vorticity(vc, i, j)
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, re, vor_eps, scheme):
        boundary_condition = get_boundary_condition(num, resolution, True)
        vorticity_confinement = (
            VorticityConfinement(boundary_condition, dt, vor_eps) if vor_eps is not None else None
        )
        pressure_updater = RedBlackSorPressureUpdater(
            boundary_condition, dt, relaxation_factor=1.3, n_iter=2
        )

        if scheme == "cip":
            solver = CipMacSolver(
                boundary_condition, pressure_updater, dt, re, vorticity_confinement
            )
        elif scheme == "upwind":
            solver = MacSolver(
                boundary_condition, pressure_updater, advect_upwind, dt, re, vorticity_confinement
            )
        elif scheme == "kk":
            solver = MacSolver(
                boundary_condition,
                pressure_updater,
                advect_kk_scheme,
                dt,
                re,
                vorticity_confinement,
            )

        return FluidSimulator(solver)


@ti.data_oriented
class DyeFluidSimulator(FluidSimulator):
    def get_dye_field(self):
        self._to_dye(self.rgb_buf, self._solver.get_fields()[2])
        return self.rgb_buf

    @ti.kernel
    def _to_dye(self, rgb_buf: ti.template(), dye: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = dye[i, j]
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, re, vor_eps, scheme):
        boundary_condition = get_boundary_condition(num, resolution, False)
        vorticity_confinement = (
            VorticityConfinement(boundary_condition, dt, vor_eps) if vor_eps is not None else None
        )
        pressure_updater = RedBlackSorPressureUpdater(
            boundary_condition, dt, relaxation_factor=1.3, n_iter=2
        )

        if scheme == "cip":
            solver = DyeCipMacSolver(
                boundary_condition, pressure_updater, dt, re, vorticity_confinement
            )
        elif scheme == "upwind":
            solver = DyeMacSolver(
                boundary_condition, pressure_updater, advect_upwind, dt, re, vorticity_confinement
            )
        elif scheme == "kk":
            solver = DyeMacSolver(
                boundary_condition,
                pressure_updater,
                advect_kk_scheme,
                dt,
                re,
                vorticity_confinement,
            )

        return DyeFluidSimulator(solver)
