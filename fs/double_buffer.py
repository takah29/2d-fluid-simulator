import taichi as ti


class DoubleBuffer:
    def __init__(self, resolution: tuple[int, int], n_channel: int) -> None:
        if n_channel == 1:
            self.current = ti.field(ti.f32, shape=resolution)
            self.next = ti.field(ti.f32, shape=resolution)
        else:
            self.current = ti.Vector.field(n_channel, ti.f32, shape=resolution)
            self.next = ti.Vector.field(n_channel, ti.f32, shape=resolution)

    def swap(self) -> None:
        self.current, self.next = self.next, self.current

    def reset(self) -> None:
        self.current.fill(0)
        self.next.fill(0)
