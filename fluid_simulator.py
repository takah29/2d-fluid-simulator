import taichi as ti

from boundary_condition import (
    create_boundary_condition1,
    create_boundary_condition2,
    create_boundary_condition3,
)
from advection import advect, advect_upwind, advect_kk_scheme, advect_eno
from updater import MacUpdater, FsUpdater
from visualization import visualize_norm


@ti.data_oriented
class FluidSimulator:
    def __init__(self, updater):
        self._updater = updater
        self.rgb_buf = ti.Vector.field(3, float, shape=updater._resolution)  # image buffer

    def step(self):
        self._updater.update()

    def get_buffer(self):
        self._to_buffer(self.rgb_buf, *self._updater.get_fields())
        return self.rgb_buf

    @ti.kernel
    def _to_buffer(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.05 * visualize_norm(vc[i, j])
            rgb_buf[i, j].x += 0.001 * pc[i, j]
            if self._updater.is_wall(i, j):
                rgb_buf[i, j].x = 0.5
                rgb_buf[i, j].y = 0.5
                rgb_buf[i, j].z = 0.7

    @staticmethod
    def create(num, resolution, dt, re):
        if num == 2:
            boundary_condition = create_boundary_condition2(resolution)
        elif num == 3:
            boundary_condition = create_boundary_condition3(resolution)
        else:
            boundary_condition = create_boundary_condition1(resolution)

        updater = MacUpdater(boundary_condition, advect_kk_scheme, dt, re, 2)
        return FluidSimulator(updater)
