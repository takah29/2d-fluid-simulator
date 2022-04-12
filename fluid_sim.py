import taichi as ti

from visualize import norm_visualize
from boundary_condition import BoundaryCondition1, BoundaryCondition2


class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(float, shape=(2 * resolution, resolution))
            self.next = ti.field(float, shape=(2 * resolution, resolution))
        else:
            self.current = ti.Vector.field(n_channel, float, shape=(2 * resolution, resolution))
            self.next = ti.Vector.field(n_channel, float, shape=(2 * resolution, resolution))

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class FluidSimulator:
    def __init__(self, resolution, dt=0.01, Re=1.0, p_iter=2):
        self._resolution = resolution
        self.dt = dt
        self.Re = Re
        self.v = DoubleBuffers(resolution, 2)  # velocities
        self.p = DoubleBuffers(resolution, 1)  # pressure
        self.buf = DoubleBuffers(resolution, 3)  # image buffer
        self.bc = None

        self.p_iter = p_iter

        # initial condition
        self.v.current.fill(ti.Vector([0.4, 0.0]))

    def step(self):
        self.bc.calc(self.v.current)
        self._update_velocities(self.v.current, self.v.next, self.p.current)
        self.v.swap()

        self.bc.calc(self.v.current)
        for _ in range(self.p_iter):
            self._update_pressures(self.p.current, self.p.next, self.v.current)
            self.p.swap()

        self._to_buffer(self.buf.current, self.v.current, self.p.current)

    def set_boundary_condition(self, boundary_condition):
        self.bc = boundary_condition

    def get_buffer(self):
        return self.buf.current

    @ti.kernel
    def _update_velocities(self, vc: ti.template(), vn: ti.template(), pc: ti.template()):
        for i, j in vn:
            vn[i, j] = vc[i, j] + self.dt * (
                -vc[i, j].x * self._diff_x(vc, i, j)
                - vc[i, j].y * self._diff_y(vc, i, j)
                - ti.Vector(
                    [
                        self._diff_x(pc, i, j),
                        self._diff_y(pc, i, j),
                    ]
                )
                + (self._diff2_x(vc, i, j) + self._diff2_y(vc, i, j)) / self.Re
            )

    @ti.kernel
    def _update_pressures(self, pc: ti.template(), pn: ti.template(), vc: ti.template()):
        for i, j in pn:
            pn[i, j] = (
                (
                    self._sample(pc, i + 1, j)
                    + self._sample(pc, i - 1, j)
                    + self._sample(pc, i, j + 1)
                    + self._sample(pc, i, j - 1)
                )
                - (self._diff_x(vc, i, j).x + self._diff_y(vc, i, j).y) / self.dt
                + self._diff_x(vc, i, j).x ** 2
                + self._diff_y(vc, i, j).y ** 2
                + 2 * self._diff_y(vc, i, j).x * self._diff_x(vc, i, j).y
            ) * 0.25

    @ti.kernel
    def _to_buffer(self, bufc: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in bufc:
            bufc[i, j] = norm_visualize(vc[i, j])
            if self.bc.bc_mask[i, j] == 1:
                bufc[i, j].x = 0.5
                bufc[i, j].y = 0.5
                bufc[i, j].z = 0.7

    @ti.func
    def _sample(self, field, i, j):
        i = max(0, min(2 * self._resolution - 1, i))
        j = max(0, min(self._resolution - 1, j))
        idx = ti.Vector([int(i), int(j)])
        return field[idx]

    @ti.func
    def _diff_x(self, field, i, j):
        return 0.5 * (self._sample(field, i + 1, j) - self._sample(field, i - 1, j))

    @ti.func
    def _diff_y(self, field, i, j):
        return 0.5 * (self._sample(field, i, j + 1) - self._sample(field, i, j - 1))

    @ti.func
    def _diff2_x(self, field, i, j):
        return (
            self._sample(field, i + 1, j)
            - 2.0 * self._sample(field, i, j)
            + self._sample(field, i - 1, j)
        )

    @ti.func
    def _diff2_y(self, field, i, j):
        return (
            self._sample(field, i, j + 1)
            - 2.0 * self._sample(field, i, j)
            + self._sample(field, i, j - 1)
        )


def main():
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    resolution = 500
    paused = False

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    fluid_sim = FluidSimulator(resolution)
    bc = BoundaryCondition2(resolution)
    fluid_sim.set_boundary_condition(bc)

    video_manager = ti.tools.VideoManager(output_dir="result", framerate=30, automatic_build=False)

    count = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused

        if not paused:
            fluid_sim.step()

        img = fluid_sim.get_buffer()
        canvas.set_image(img)
        window.show()

        # if count % 200 == 0:
        #     video_manager.write_frame(img)

        count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
