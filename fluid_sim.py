import taichi as ti

from visualize import visualize_norm, visualize_xy, visualize_hue
from boundary_condition import BoundaryCondition1, BoundaryCondition2, BoundaryCondition3


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
class FluidSimulator:
    def __init__(self, boundary_condition, dt, Re, p_iter=5):
        self.bc = boundary_condition
        self._resolution = boundary_condition.get_resolution()
        self.dt = dt
        self.Re = Re
        self.v = DoubleBuffers(self._resolution, 2)  # velocities
        self.p = DoubleBuffers(self._resolution, 1)  # pressure
        self.rgb_buf = ti.Vector.field(3, float, shape=self._resolution)  # image buffer

        self.p_iter = p_iter

        # initial condition
        self.v.current.fill(ti.Vector([0.4, 0.0]))
        self.bc.calc(self.v.current, self.p.current)

    def step(self):
        self.bc.calc(self.v.current, self.p.current)
        self._update_velocities(self.v.current, self.v.next, self.p.current)
        self.v.swap()

        self.bc.calc(self.v.current, self.p.current)
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
    def _advect(self, vc, i, j):
        return vc[i, j].x * self._diff_x(vc, i, j) + vc[i, j].y * self._diff_y(vc, i, j)

    @ti.func
    def _advect_upwind(self, vc, i, j):
        """Upwind differencing

        http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#tac8e468
        """
        k = 0
        if vc[i, j].x < 0.0:
            k = i
        else:
            k = i - 1
        a = vc[i, j].x * (self._sample(vc, k + 1, j) - self._sample(vc, k, j))

        if vc[i, j].y < 0.0:
            k = j
        else:
            k = j - 1

        b = vc[i, j].y * (self._sample(vc, i, k + 1) - self._sample(vc, i, k))

        return a + b

    @ti.func
    def _advect_kk_scheme(self, vc, i, j):
        """Kawamura-Kuwabara Scheme

        http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#y2dbc484
        """
        a = b = ti.Vector([0.0, 0.0])
        if vc[i, j].x < 0:
            a = (
                vc[i, j].x
                * (
                    -2 * self._sample(vc, i + 2, j)
                    + 10 * self._sample(vc, i + 1, j)
                    - 9 * self._sample(vc, i, j)
                    + 2 * self._sample(vc, i - 1, j)
                    - self._sample(vc, i - 2, j)
                )
                / 6
            )
        else:
            a = (
                vc[i, j].x
                * (
                    self._sample(vc, i + 2, j)
                    - 2 * self._sample(vc, i + 1, j)
                    + 9 * self._sample(vc, i, j)
                    - 10 * self._sample(vc, i - 1, j)
                    + 2 * self._sample(vc, i - 2, j)
                )
                / 6
            )

        if vc[i, j].y < 0:
            b = (
                vc[i, j].y
                * (
                    -2 * self._sample(vc, i, j + 2)
                    + 10 * self._sample(vc, i, j + 1)
                    - 9 * self._sample(vc, i, j)
                    + 2 * self._sample(vc, i, j - 1)
                    - self._sample(vc, i, j - 2)
                )
                / 6
            )
        else:
            b = (
                vc[i, j].y
                * (
                    self._sample(vc, i, j + 2)
                    - 2 * self._sample(vc, i, j + 1)
                    + 9 * self._sample(vc, i, j)
                    - 10 * self._sample(vc, i, j - 1)
                    + 2 * self._sample(vc, i, j - 2)
                )
                / 6
            )

        return a + b

    @ti.func
    def _advect_eno(self, vc, i, j):
        """ENO(Essentially Non-Oscillatory polynomial interpolation)

        http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1#ufe6b856
        """
        c2 = c3 = ti.Vector([0.0, 0.0])
        k = k_star = 0
        # calc advect_x
        # first order
        if vc[i, j].x < 0.0:
            k = i
        else:
            k = i - 1
        Dk_1 = self._sample(vc, k + 1, j) - self._sample(vc, k, j)
        qx1 = Dk_1

        # second order
        Dk_2 = self._diff2_x(vc, k, j) / 2.0
        Dk1_2 = self._diff2_x(vc, k + 1, j) / 2.0

        if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x):
            c2 = Dk_2
        else:
            c2 = Dk1_2

        qx2 = c2 * (2 * (i - k) - 1)

        # third order
        if ti.abs(Dk_2.x) <= ti.abs(Dk1_2.x):
            k_star = k - 1
        else:
            k_star = k
        Dk_star_3 = self._diff3_x(vc, k_star, j) / 6.0
        Dk_star1_3 = self._diff3_x(vc, k_star + 1, j) / 6.0

        if ti.abs(Dk_star_3.x) <= ti.abs(Dk_star1_3.x):
            c3 = Dk_star_3
        else:
            c3 = Dk_star1_3

        qx3 = c3 * (3 * (i - k_star) ** 2 - 6 * (i - k_star) + 2)

        advect_x = vc[i, j].x * (qx1 + qx2 + qx3)

        # calc advect_y
        # first order
        if vc[i, j].y < 0.0:
            k = j
        else:
            k = j - 1
        Dk_1 = self._sample(vc, i, k + 1) - self._sample(vc, i, k)
        qy1 = Dk_1

        # second order
        Dk_2 = self._diff2_y(vc, i, k) / 2.0
        Dk1_2 = self._diff2_y(vc, i, k + 1) / 2.0
        if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y):
            c2 = Dk_2
        else:
            c2 = Dk1_2
        qy2 = c2 * (2 * (j - k) - 1)

        # third order
        if ti.abs(Dk_2.y) <= ti.abs(Dk1_2.y):
            k_star = k - 1
        else:
            k_star = k
        Dk_star_3 = self._diff3_y(vc, i, k_star) / 6.0
        Dk_star1_3 = self._diff3_y(vc, i, k_star + 1) / 6.0
        if ti.abs(Dk_star_3.y) <= ti.abs(Dk_star1_3.y):
            c3 = Dk_star_3
        else:
            c3 = Dk_star1_3
        qy3 = c3 * (3 * (j - k_star) ** 2 - 6 * (j - k_star) + 2)

        advect_y = vc[i, j].y * (qy1 + qy2 + qy3)

        return advect_x + advect_y

    @ti.func
    def _sample(self, field, i, j):
        i = max(0, min(self._resolution[0] - 1, i))
        j = max(0, min(self._resolution[1] - 1, j))
        idx = ti.Vector([int(i), int(j)])
        return field[idx]

    @ti.func
    def _diff_x(self, field, i, j):
        """Central Difference x
        """
        return 0.5 * (self._sample(field, i + 1, j) - self._sample(field, i - 1, j))

    @ti.func
    def _diff_y(self, field, i, j):
        """Central Difference y
        """
        return 0.5 * (self._sample(field, i, j + 1) - self._sample(field, i, j - 1))

    @ti.func
    def _diff2_x(self, field, i, j):
        return self._sample(field, i + 1, j) - 2.0 * field[i, j] + self._sample(field, i - 1, j)

    @ti.func
    def _diff2_y(self, field, i, j):
        return self._sample(field, i, j + 1) - 2.0 * field[i, j] + self._sample(field, i, j - 1)

    @ti.func
    def _diff3_x(self, field, i, j):
        return (
            self._sample(field, i + 2, j)
            - 3.0 * self._sample(field, i + 1, j)
            + 3.0 * field[i, j]
            - self._sample(field, i - 1, j)
        )

    @ti.func
    def _diff3_y(self, field, i, j):
        return (
            self._sample(field, i, j + 2)
            - 3.0 * self._sample(field, i, j + 1)
            + 3.0 * field[i, j]
            - self._sample(field, i, j - 1)
        )


def main():
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    resolution = 400
    paused = False

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    bc = BoundaryCondition3(resolution)
    fluid_sim = FluidSimulator(bc, 0.01, 1000.0, 2)

    # video_manager = ti.tools.VideoManager(output_dir="result", framerate=60, automatic_build=False)

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
