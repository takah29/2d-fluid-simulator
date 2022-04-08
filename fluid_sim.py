import numpy as np

import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)


class DoubleBuffers:
    def __init__(self, resolution, n_channel):
        if n_channel == 1:
            self.current = ti.field(float, shape=(resolution, resolution))
            self.next = ti.field(float, shape=(resolution, resolution))
        else:
            self.current = ti.Vector.field(n_channel, float, shape=(resolution, resolution))
            self.next = ti.Vector.field(n_channel, float, shape=(resolution, resolution))

    def swap(self):
        self.current, self.next = self.next, self.current

    def reset(self):
        self.current.fill(0)
        self.next.fill(0)


@ti.data_oriented
class FluidSimulator:
    def __init__(self, resolution, dt=0.03, Re=1.54, p_iter=50):
        self._resolution = resolution
        self.dt = dt
        self.Re = Re
        self.v = DoubleBuffers(resolution, 2)  # velocities
        self.p = DoubleBuffers(resolution, 1)  # pressure
        self.buf = DoubleBuffers(resolution, 3)
        self.bc = None

        self.p_iter = p_iter
        self.g = ti.Vector([0, -9.8])

    def step(self):
        self._set_bc()
        self._update_velocities()
        print(self.v.current[1, 100])

        self.v.swap()

        for _ in range(self.p_iter):
            self._update_pressures()
            self.p.swap()

        self._to_buffer()

    def set_boundary_condition(self, boundary_condition):
        self.bc = boundary_condition

    def get_buffer(self):
        return self.buf.current

    @ti.kernel
    def _update_velocities(self):
        for i, j in self.v.next:
            self.v.next[i, j] = self.v.current[i, j] + self.dt * (
                -self.v.current[i, j].x * self._diff_x(self.v.current, i, j)
                - self.v.current[i, j].y * self._diff_y(self.v.current, i, j)
                - ti.Vector(
                    [
                        self._diff_x(self.p.current, i, j),
                        self._diff_y(self.p.current, i, j),
                    ]
                )
                + (self._diff2_x(self.v.current, i, j) + self._diff2_y(self.v.current, i, j))
                / self.Re
            )

    @ti.kernel
    def _update_pressures(self):
        for i, j in self.p.next:
            self.p.next[i, j] = (
                (
                    self._sample(self.p.current, i + 1, j)
                    + self._sample(self.p.current, i - 1, j)
                    + self._sample(self.p.current, i, j + 1)
                    + self._sample(self.p.current, i, j - 1)
                )
                - (self._diff_x(self.v.current, i, j).x + self._diff_y(self.v.current, i, j).y)
                / self.dt
                + self._diff_x(self.v.current, i, j).x ** 2
                + self._diff_y(self.v.current, i, j).y ** 2
                + 2 * self._diff_y(self.v.current, i, j).x * self._diff_x(self.v.current, i, j).y
            ) * 0.25

    @ti.kernel
    def _set_bc(self):
        for i, j in self.v.current:
            if (self.bc[i, j] >= ti.Vector([0.0, 0.0])).all():
                self.v.current[i, j] = self.bc[i, j]
                self.v.next[i, j] = self.bc[i, j]

    @ti.kernel
    def _to_buffer(self):
        for i, j in self.buf.current:
            self.buf.current[i, j].x = 256.0 * self.v.current[i, j].x
            self.buf.current[i, j].y = 256.0 * self.v.current[i, j].y
            self.buf.current[i, j].z = 256.0 * self.p.current[i, j]

    @ti.func
    def _sample(self, field, i, j):
        idx = ti.Vector([int(i), int(j)])
        idx = max(0, min(self._resolution - 1, idx))
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


def create_bc(resolution):
    bc = -np.ones((resolution, resolution, 2))

    # 流入部、流出部の設定
    bc[0, :] = np.array([1.0, 0.0])
    bc[-1, :] = np.array([1.0, 0.0])

    # 壁の設定
    bc[:, 0] = np.array([0.0, 0.0])
    bc[:, -1] = np.array([0.0, 0.0])
    size = resolution // 6
    bc[
        resolution // 2 - 2 * size : resolution // 2,
        resolution // 2 - size : resolution // 2 + size,
    ] = np.array([0.0, 0.0])

    bc_field = ti.Vector.field(2, float, shape=(resolution, resolution))
    bc_field.from_numpy(bc)

    return bc_field


def main():
    resolution = 1000
    max_fps = 60

    paused = False
    debug = False

    window = ti.ui.Window("Fluid Simulation", (resolution, resolution), vsync=True)
    canvas = window.get_canvas()

    fluid_sim = FluidSimulator(resolution)
    bc = create_bc(resolution)
    fluid_sim.set_boundary_condition(bc)

    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused
            elif e.key == "d":
                debug = not debug

        if not paused:
            fluid_sim.step()

        canvas.set_image(fluid_sim.get_buffer())
        window.show()


if __name__ == "__main__":
    main()
