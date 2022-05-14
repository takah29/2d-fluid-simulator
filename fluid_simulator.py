import sys

import taichi as ti


from boundary_condition import (
    create_boundary_condition1,
    create_boundary_condition2,
    create_boundary_condition3,
    create_boundary_condition4,
    create_dyes_boundary_condition1,
    create_dyes_boundary_condition2,
    create_dyes_boundary_condition3,
    create_dyes_boundary_condition4,
)
from advection import advect, advect_upwind, advect_kk_scheme
from solver import MacSolver, DyesMacSolver, CipMacSolver, DyesCipMacSolver
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
            rgb_buf[i, j] = 0.05 * visualize_norm(vc[i, j])
            rgb_buf[i, j] += 0.002 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_pressure(self, rgb_buf: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.02 * visualize_pressure(pc[i, j])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @ti.kernel
    def _to_vorticity(self, rgb_buf: ti.template(), vc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.5 * visualize_vorticity(vc, i, j)
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, re):
        if num == 1:
            boundary_condition = create_boundary_condition1(resolution)
        elif num == 2:
            boundary_condition = create_boundary_condition2(resolution)
        elif num == 3:
            boundary_condition = create_boundary_condition3(resolution)
        elif num == 4:
            boundary_condition = create_boundary_condition4(resolution)
        else:
            raise NotImplementedError

        solver = CipMacSolver(boundary_condition, dt, re, 2)
        return FluidSimulator(solver)


@ti.data_oriented
class DyesFluidSimulator(FluidSimulator):
    def get_dye_field(self):
        self._to_dye(self.rgb_buf, *self._solver.get_fields())
        return self.rgb_buf

    @ti.kernel
    def _to_dye(
        self, rgb_buf: ti.template(), v: ti.template(), p: ti.template(), dyes: ti.template()
    ):
        for i, j in rgb_buf:
            rgb_buf[i, j] = dyes[i, j]
            if self._solver.is_wall(i, j):
                rgb_buf[i, j] = self._wall_color

    @staticmethod
    def create(num, resolution, dt, re):
        if num == 1:
            boundary_condition = create_dyes_boundary_condition1(resolution)
        elif num == 2:
            boundary_condition = create_dyes_boundary_condition2(resolution)
        elif num == 3:
            boundary_condition = create_dyes_boundary_condition3(resolution)
        elif num == 4:
            boundary_condition = create_dyes_boundary_condition4(resolution)
        else:
            raise NotImplementedError

        solver = DyesCipMacSolver(boundary_condition, dt, re, 2)
        return DyesFluidSimulator(solver)
