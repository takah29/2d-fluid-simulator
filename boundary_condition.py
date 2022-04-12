import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition1:
    def __init__(self, resolution):
        self._resolution = resolution
        self._const_bc, self.bc_mask = BoundaryCondition1._set_const_bc(resolution)

    @staticmethod
    def _set_const_bc(resolution):
        bc = np.zeros((2 * resolution, resolution, 2))
        # 1: 壁, 2: 定常流
        bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)
        size = resolution // 18

        # 流入部、流出部の設定
        bc[0, :] = np.array([0.4, 0.0])
        bc_mask[0, :] = 2
        bc[0, resolution // 2 - 2 * size : resolution // 2 + 2 * size] = np.array([4.0, 0.0])
        bc_mask[0, resolution // 2 - 2 * size : resolution // 2 + 2 * size] = 2

        # 壁の設定
        bc[:, 0] = np.array([0.0, 0.0])
        bc_mask[:, 0] = 1
        bc[:, -1] = np.array([0.0, 0.0])
        bc_mask[:, -1] = 1
        bc[
            resolution // 2 - 2 * size : resolution // 2,
            resolution // 2 - size : resolution // 2 + size,
        ] = np.array([0.0, 0.0])
        bc_mask[
            resolution // 2 - 2 * size : resolution // 2,
            resolution // 2 - size : resolution // 2 + size,
        ] = 1

        bc_field = ti.Vector.field(2, float, shape=(2 * resolution, resolution))
        bc_field.from_numpy(bc)
        bc_mask_field = ti.field(ti.types.u8, shape=(2 * resolution, resolution))
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_mask_field

    @ti.kernel
    def calc(self, vc: ti.template()):
        for i, j in vc:
            if self.bc_mask[i, j] > 0:
                vc[i, j] = self._const_bc[i, j]
            if i == (2 * self._resolution - 1) and j != 0 and j != self._resolution - 1:
                vc[i, j] = vc[i - 1, j]
