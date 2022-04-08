import numpy as np

import taichi as ti

ti.init(arch=ti.cuda)


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


class MouseDataGen:
    def __init__(self, resolution):
        self.prev_mouse = None
        self.prev_color = None

        self.resolution = resolution

    def __call__(self, window):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32) * self.resolution
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


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
        self.force_radius = resolution / 2.0
        self.f_strength = 10.0
        self.g = ti.Vector([0, -9.8])
        self.dye_decay = 1 - 1.0 / 120.0

    def step(self, mouse_data: ti.ext_arr()):
        self._set_bc()
        self._update_velocities()
        # self._update_buffer()

        self.v.swap()
        self._set_bc()
        # self.buf.swap()

        # self._apply_impulse(mouse_data)
        for _ in range(self.p_iter):
            self._update_pressures()
            self.p.swap()

        self.to_buffer()

    @ti.kernel
    def to_buffer(self):
        for i, j in self.buf.current:
            self.buf.current[i, j].x = 256.0 * self.v.current[i, j].x
            self.buf.current[i, j].y = 256.0 * self.v.current[i, j].y
            self.buf.current[i, j].z = 256.0 * self.p.current[i, j]

    def set_boundary_condition(self, boundary_condition):
        self.bc = boundary_condition

    # @ti.kernel
    # def _apply_impulse(self, imp_data: ti.ext_arr()):
    #     g_dir = self.g * 300
    #     for i, j in self.v.current:
    #         omx, omy = imp_data[2], imp_data[3]
    #         mdir = ti.Vector([imp_data[0], imp_data[1]])
    #         dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
    #         d2 = dx * dx + dy * dy
    #         # dv = F * dt
    #         factor = ti.exp(-d2 / self.force_radius)

    #         dc = self.buf.current[i, j]
    #         a = dc.norm()

    #         momentum = (mdir * self.f_strength * factor + g_dir * a / (1 + a)) * self.dt

    #         v = self.v.current[i, j]
    #         self.v.current[i, j] = v + momentum
    #         # add dye
    #         if mdir.norm() > 0.5:
    #             dc += ti.exp(-d2 * (4 / (self._resolution / 15) ** 2)) * ti.Vector(
    #                 [imp_data[4], imp_data[5], imp_data[6]]
    #             )

    #         self.buf.current[i, j] = dc

    def get_buffer(self):
        return self.buf.current

    @ti.func
    def _sample(self, field, i, j):
        idx = ti.Vector([int(i), int(j)])
        idx = max(0, min(self._resolution - 1, idx))
        return field[idx]

    # @ti.func
    # def _lerp(self, p, q, t):
    #     # t: [0.0, 1.0]
    #     return p + t * (q - p)

    # @ti.func
    # def _bilerp(self, field, p):
    #     u, v = p
    #     s, t = u - 0.5, v - 0.5
    #     # floor
    #     iu, iv = ti.floor(s), ti.floor(t)
    #     # fract
    #     fu, fv = s - iu, t - iv
    #     a = self._sample(field, iu, iv)
    #     b = self._sample(field, iu + 1, iv)
    #     c = self._sample(field, iu, iv + 1)
    #     d = self._sample(field, iu + 1, iv + 1)
    #     return self._lerp(self._lerp(a, b, fu), self._lerp(c, d, fu), fv)

    # # 3rd order Runge-Kutta
    # @ti.func
    # def _backtrace(self, vf: ti.template(), p):
    #     v1 = self._bilerp(vf, p)
    #     p1 = p - 0.5 * self.dt * v1
    #     v2 = self._bilerp(vf, p1)
    #     p2 = p - 0.75 * self.dt * v2
    #     v3 = self._bilerp(vf, p2)
    #     p -= self.dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    #     return p

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

    @ti.kernel
    def _set_bc(self):
        for i, j in self.v.current:
            if (self.bc[i, j] >= ti.Vector([0.0, 0.0])).all():
                self.v.current[i, j] = self.bc[i, j]

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

    # @ti.kernel
    # def _update_buffer(self):
    #     for i, j in self.buf.next:
    #         p = ti.Vector([i, j])
    #         p = self._backtrace(self.v.current, p)
    #         self.buf.next[i, j] = self._bilerp(self.buf.current, p) * self.dye_decay

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


def create_bc(resolution):
    bc = -np.ones((resolution, resolution, 2))

    # 流入部、流出部の設定
    bc[0, :] = np.array([100.0, 0.0])
    bc[-1, :] = np.array([100.0, 0.0])

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
    md_gen = MouseDataGen(resolution)

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
            mouse_data = md_gen(window)
            fluid_sim.step(mouse_data)
            print(mouse_data)

        canvas.set_image(fluid_sim.get_buffer())
        window.show()


if __name__ == "__main__":
    main()
