import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition1:
    def __init__(self, resolution):
        self._resolution = resolution
        self._const_bc = BoundaryCondition1._set_const_bc(resolution)

    @staticmethod
    def _set_const_bc(resolution):
        bc = -np.ones((2 * resolution, resolution, 2))
        size = resolution // 18

        # 流入部、流出部の設定
        bc[0, :] = np.array([0.4, 0.0])
        bc[0, resolution // 2 - 2 * size : resolution // 2 + 2 * size] = np.array([4.0, 0.0])

        # 壁の設定
        bc[:, 0] = np.array([0.0, 0.0])
        bc[:, -1] = np.array([0.0, 0.0])
        bc[
            resolution // 2 - 2 * size : resolution // 2,
            resolution // 2 - size : resolution // 2 + size,
        ] = np.array([0.0, 0.0])

        bc_field = ti.Vector.field(2, float, shape=(2 * resolution, resolution))
        bc_field.from_numpy(bc)

        return bc_field

    @ti.kernel
    def calc(self, vc: ti.template()):
        for i, j in vc:
            if (self._const_bc[i, j] >= ti.Vector([0.0, 0.0])).all():
                vc[i, j] = self._const_bc[i, j]
            if i == (2 * self._resolution - 1) and j != 0 and j != self._resolution - 1:
                vc[i, j] = vc[i - 1, j]
