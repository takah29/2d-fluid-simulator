import sys

import taichi as ti


from boundary_condition import (
    create_boundary_condition1,
    create_boundary_condition2,
    create_boundary_condition3,
    create_dyes_boundary_condition1,
    create_dyes_boundary_condition2,
    create_dyes_boundary_condition3,
)
from advection import advect, advect_upwind, advect_kk_scheme
from solver import MacSolver, DyesMacSolver, CipMacSolver, DyesCipMacSolver
from visualization import visualize_norm


@ti.data_oriented
class FluidSimulator:
    def __init__(self, solver):
        self._solver = solver
        self.rgb_buf = ti.Vector.field(3, float, shape=solver._resolution)  # image buffer

    def step(self):
        self._solver.update()

    def get_buffer(self):
        self._to_buffer(self.rgb_buf, *self._solver.get_fields())
        return self.rgb_buf

    @ti.kernel
    def _to_buffer(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.05 * visualize_norm(vc[i, j])
            rgb_buf[i, j].x += 0.001 * pc[i, j]
            if self._solver.is_wall(i, j):
                rgb_buf[i, j].x = 0.5
                rgb_buf[i, j].y = 0.7
                rgb_buf[i, j].z = 0.5

    @staticmethod
    def create(num, resolution, dt, re):
        if num == 1:
            boundary_condition = create_boundary_condition1(resolution)
        elif num == 2:
            boundary_condition = create_boundary_condition2(resolution)
        elif num == 3:
            boundary_condition = create_boundary_condition3(resolution)
        else:
            raise NotImplementedError

        solver = CipMacSolver(boundary_condition, dt, re, 2)
        return FluidSimulator(solver)


@ti.data_oriented
class DyesFluidSimulator(FluidSimulator):
    @ti.kernel
    def _to_buffer(
        self, rgb_buf: ti.template(), dyes: ti.template(), v: ti.template(), p: ti.template()
    ):
        for i, j in rgb_buf:
            rgb_buf[i, j] = dyes[i, j]
            # c = 0.001 * p[i, j]
            # rgb_buf[i, j] += ti.Vector([c, c, c])
            if self._solver.is_wall(i, j):
                rgb_buf[i, j].x = 0.5
                rgb_buf[i, j].y = 0.7
                rgb_buf[i, j].z = 0.5

    @staticmethod
    def create(num, resolution, dt, re):
        if num == 1:
            boundary_condition = create_dyes_boundary_condition1(resolution)
        elif num == 2:
            boundary_condition = create_dyes_boundary_condition2(resolution)
        elif num == 3:
            boundary_condition = create_dyes_boundary_condition3(resolution)
        else:
            raise NotImplementedError

        solver = DyesCipMacSolver(boundary_condition, advect_kk_scheme, dt, re, 2)
        return DyesFluidSimulator(solver)
