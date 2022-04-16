import taichi as ti

from visualize import visualize_norm, visualize_hue
from boundary_condition import BoundaryCondition1, BoundaryCondition2, BoundaryCondition3


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
    def __init__(self, boundary_condition, dt=0.01, Re=10000.0, p_iter=5):
        self.bc = boundary_condition
        self._resolution = boundary_condition.resolution
        self.dt = dt
        self.Re = Re
        self.v = DoubleBuffers(self._resolution, 2)  # velocities
        self.p = DoubleBuffers(self._resolution, 1)  # pressure
        self.rgb_buf = ti.Vector.field(
            3, float, shape=(2 * self._resolution, self._resolution)
        )  # image buffer

        self.p_iter = p_iter

        # initial condition
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.bc.calc(self.v.current, self.p.current)

    def step(self):
        self.bc.calc(self.v.current, self.p.current)
        self._update_velocities(self.v.current, self.v.next, self.p.current)
        self.v.swap()

        for _ in range(self.p_iter):
            self._update_pressures(self.p.current, self.p.next, self.v.current)
            self.p.swap()

        self._to_buffer(self.rgb_buf, self.v.current, self.p.current)

    def get_buffer(self):
        return self.rgb_buf

    @ti.kernel
    def _update_velocities(self, vc: ti.template(), vn: ti.template(), pc: ti.template()):
        for i, j in vn:
            if not self.bc.is_wall(i, j):
                vn[i, j] = vc[i, j] + self.dt * (
                    -self._advect_kk_scheme(vc, i, j)
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
            if not self.bc.is_wall(i, j):
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
    def _to_buffer(self, rgb_buf: ti.template(), vc: ti.template(), pc: ti.template()):
        for i, j in rgb_buf:
            rgb_buf[i, j] = 0.05 * visualize_norm(vc[i, j])
            # rgb_buf[i, j].x += 0.001 * pc[i, j]
            if self.bc.is_wall(i, j):
                rgb_buf[i, j].x = 0.5
                rgb_buf[i, j].y = 0.5
                rgb_buf[i, j].z = 0.7

    @ti.func
    def _advect(self, field, i, j):
        return field[i, j].x * self._diff_x(field, i, j) + field[i, j].y * self._diff_y(field, i, j)

    @ti.func
    def _advect_kk_scheme(self, field, i, j):
        """Kawamura-Kuwabara Scheme

        http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#y2dbc484
        """
        a = b = ti.Vector([0.0, 0.0])
        if field[i, j].x < 0:
            a = (
                field[i, j].x
                * (
                    -2 * self._sample(field, i + 2, j)
                    + 10 * self._sample(field, i + 1, j)
                    - 9 * self._sample(field, i, j)
                    + 2 * self._sample(field, i - 1, j)
                    - self._sample(field, i - 2, j)
                )
                / 6
            )
        else:
            a = (
                field[i, j].x
                * (
                    self._sample(field, i + 2, j)
                    - 2 * self._sample(field, i + 1, j)
                    + 9 * self._sample(field, i, j)
                    - 10 * self._sample(field, i - 1, j)
                    + 2 * self._sample(field, i - 2, j)
                )
                / 6
            )

        if field[i, j].y < 0:
            b = (
                field[i, j].y
                * (
                    -2 * self._sample(field, i, j + 2)
                    + 10 * self._sample(field, i, j + 1)
                    - 9 * self._sample(field, i, j)
                    + 2 * self._sample(field, i, j - 1)
                    - self._sample(field, i, j - 2)
                )
                / 6
            )
        else:
            b = (
                field[i, j].y
                * (
                    self._sample(field, i, j + 2)
                    - 2 * self._sample(field, i, j + 1)
                    + 9 * self._sample(field, i, j)
                    - 10 * self._sample(field, i, j - 1)
                    + 2 * self._sample(field, i, j - 2)
                )
                / 6
            )

        return a + b

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

    resolution = 400
    paused = False

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    bc = BoundaryCondition3(resolution)
    fluid_sim = FluidSimulator(bc)

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

        # if count % 50 == 0:
        #     video_manager.write_frame(img)

        count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
