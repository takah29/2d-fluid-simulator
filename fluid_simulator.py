import taichi as ti

from advection import advect_kk_scheme, advect_upwind
from boundary_condition import (
    create_boundary_condition1,
    create_boundary_condition2,
    create_boundary_condition3,
    create_boundary_condition4,
)
from solver import CipMacSolver, DyeCipMacSolver, DyeMacSolver, MacSolver
from visualization import visualize_norm, visualize_pressure, visualize_vorticity


@ti.data_oriented
class FluidSimulator:
    def __init__(self, solver):
        self._solver = solver
        self.rgb_buf = ti.Vector.field(3, float, shape=solver._resolution)  # image buffer
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
            rgb_buf[i, j] = 0.02 * visualize_norm(vc[i, j])
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
        if num == 1:
            boundary_condition = create_boundary_condition1(resolution, no_dye=True)
        elif num == 2:
            boundary_condition = create_boundary_condition2(resolution, no_dye=True)
        elif num == 3:
            boundary_condition = create_boundary_condition3(resolution, no_dye=True)
        elif num == 4:
            boundary_condition = create_boundary_condition4(resolution, no_dye=True)
        else:
            raise NotImplementedError

        if scheme == "cip":
            solver = CipMacSolver(boundary_condition, dt, re, 2, vor_eps)
        elif scheme == "upwind":
            solver = MacSolver(boundary_condition, advect_upwind, dt, re, 2, vor_eps)
        elif scheme == "kk":
            solver = MacSolver(boundary_condition, advect_kk_scheme, dt, re, 2, vor_eps)

        return FluidSimulator(solver)


@ti.data_oriented
class DyeFluidSimulator(FluidSimulator):
    def get_dye_field(self):
        self._to_dye(self.rgb_buf, *self._solver.get_fields())
        return self.rgb_buf

    @ti.kernel
    def _to_dye(
        self, rgb_buf: ti.template(), v: ti.template(), p: ti.template(), dye: ti.template()
    ):
        for i, j in rgb_buf:
            rgb_buf[i, j] = dye[i, j]
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, re, vor_eps, scheme):
        if num == 1:
            boundary_condition = create_boundary_condition1(resolution, no_dye=False)
        elif num == 2:
            boundary_condition = create_boundary_condition2(resolution, no_dye=False)
        elif num == 3:
            boundary_condition = create_boundary_condition3(resolution, no_dye=False)
        elif num == 4:
            boundary_condition = create_boundary_condition4(resolution, no_dye=False)
        else:
            raise NotImplementedError

        if scheme == "cip":
            solver = DyeCipMacSolver(boundary_condition, dt, re, 2, vor_eps)
        elif scheme == "upwind":
            solver = DyeMacSolver(boundary_condition, advect_upwind, dt, re, 2, vor_eps)
        elif scheme == "kk":
            solver = DyeMacSolver(boundary_condition, advect_kk_scheme, dt, re, 2, vor_eps)

        return DyeFluidSimulator(solver)
