import taichi as ti

from boundary_condition import BoundaryCondition, DyeBoundaryCondition
from differentiation import diff_x, diff_y


@ti.data_oriented
class VorticityConfinement:
    def __init__(
        self,
        boundary_condition: BoundaryCondition | DyeBoundaryCondition,
        dt: float,
        dx: float,
        weight: float,
    ) -> None:
        self._bc = boundary_condition
        self.dt = dt
        self.dx = dx
        self.weight = weight

        self._resolution = boundary_condition.get_resolution()

        self.vorticity = ti.field(ti.f32, shape=self._resolution)
        self.vorticity_abs = ti.field(ti.f32, shape=self._resolution)

    @ti.kernel
    def _calc_vorticity(self, vc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN202
        for i, j in self.vorticity:
            if self._bc.is_fluid_domain(i, j):
                self.vorticity[i, j] = diff_x(vc, i, j, self.dx).y - diff_y(vc, i, j, self.dx).x  # type: ignore[reportAttributeAccessIssue]
                self.vorticity_abs[i, j] = ti.abs(self.vorticity[i, j])

    @ti.kernel
    def _add_vorticity(  # noqa: ANN202
        self,
        vn: ti.template(),  # type: ignore[valid-type]
        vc: ti.template(),  # type: ignore[valid-type]
    ):
        for i, j in vn:
            if self._bc.is_fluid_domain(i, j):
                vn[i, j] = vc[i, j] + self.dt * self.weight * self._vorticity_vec(i, j)

    @ti.func
    def _vorticity_vec(self, i: int, j: int) -> ti.Vector:
        vorticity_grad_vec = ti.Vector(
            [diff_x(self.vorticity_abs, i, j, self.dx), diff_y(self.vorticity_abs, i, j, self.dx)]
        )
        vorticity_grad_vec = vorticity_grad_vec / vorticity_grad_vec.norm()
        vorticity_vec = (
            ti.Vector([vorticity_grad_vec.y, -vorticity_grad_vec.x]) * self.vorticity[i, j]  # type: ignore[reportOptionalOperand]
        )

        # 発散する可能性があるのでクランプする
        return ti.max(ti.min(vorticity_vec, 0.1), -0.1)

    def apply(self, v) -> None:
        self._calc_vorticity(v.current)
        self._add_vorticity(v.next, v.current)  # v.nextのみ更新する
