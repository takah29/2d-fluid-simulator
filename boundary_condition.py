import numpy as np
import taichi as ti

np.random.seed(123)


def to_field(bc, bc_mask):
    bc_field = ti.Vector.field(2, ti.types.f64, shape=(bc.shape[0], bc.shape[1]))
    bc_field.from_numpy(bc)
    bc_mask_field = ti.field(ti.types.u8, shape=(bc_mask.shape[0], bc_mask.shape[1]))
    bc_mask_field.from_numpy(bc_mask)

    return bc_field, bc_mask_field


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

        # 流入部の設定
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

        return to_field(bc, bc_mask)

    @ti.kernel
    def calc(self, vc: ti.template()):
        for i, j in vc:
            if self.bc_mask[i, j] > 0:
                vc[i, j] = self._const_bc[i, j]
            if i == (2 * self._resolution - 1) and 0 <= j < self._resolution:
                vc[i, j] = ti.Vector([ti.min(ti.max(vc[i - 1, j].x, 0.0), 10.0), vc[i - 1, j].y])


@ti.data_oriented
class BoundaryCondition2:
    def __init__(self, resolution):
        self._resolution = resolution
        self._const_bc, self.bc_mask = BoundaryCondition2._set_const_bc(resolution)

    @staticmethod
    def _set_const_bc(resolution):
        bc = np.zeros((2 * resolution, resolution, 2))
        bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

        # 壁の設定
        size = resolution // 32  # 壁幅
        bc[0, :] = np.array([0.0, 0.0])
        bc_mask[0, :] = 1
        bc[-1, :] = np.array([0.0, 0.0])
        bc_mask[-1, :] = 1
        bc[:, 0] = np.array([0.0, 0.0])
        bc_mask[:, 0] = 1
        bc[:, -1] = np.array([0.0, 0.0])
        bc_mask[:, -1] = 1

        x_point = 2 * resolution // 5
        y_point = resolution // 2
        # 左
        bc[x_point - size : x_point + size, y_point:] = np.array([0.0, 0.0])
        bc_mask[x_point - size : x_point + size, y_point:] = 1
        # 真ん中左
        bc[2 * x_point - size : 2 * x_point + size, 0:y_point] = np.array([0.0, 0.0])
        bc_mask[2 * x_point - size : 2 * x_point + size, 0:y_point:] = 1
        # 真ん中右
        bc[3 * x_point - size : 3 * x_point + size, y_point:] = np.array([0.0, 0.0])
        bc_mask[3 * x_point - size : 3 * x_point + size, y_point:] = 1
        # 右
        bc[4 * x_point - size : 4 * x_point + size, 0:y_point] = np.array([0.0, 0.0])
        bc_mask[4 * x_point - size : 4 * x_point + size, 0:y_point] = 1

        # 流入部、流出部の設定
        y_point = resolution // 6
        bc[0, 2 * y_point : 4 * y_point] = np.array([4.0, 0.0])
        bc_mask[0, 2 * y_point : 4 * y_point] = 2
        bc[-1, 2 * y_point : 4 * y_point] = np.array([4.0, 0.0])
        bc_mask[-1, 2 * y_point : 4 * y_point] = 2

        return to_field(bc, bc_mask)

    @ti.kernel
    def calc(self, vc: ti.template()):
        for i, j in vc:
            if self.bc_mask[i, j] > 0:
                vc[i, j] = self._const_bc[i, j]
            if (
                i == (2 * self._resolution - 1)
                and 2 * self._resolution // 6 <= j < 4 * self._resolution // 6
            ):
                vc[i, j] = ti.Vector([ti.min(ti.max(vc[i - 1, j].x, 0.0), 10.0), vc[i - 1, j].y])


@ti.data_oriented
class BoundaryCondition3:
    def __init__(self, resolution):
        self._resolution = resolution
        self._const_bc, self.bc_mask = BoundaryCondition3._set_const_bc(resolution)

    @staticmethod
    def _set_const_bc(resolution):
        bc = np.zeros((2 * resolution, resolution, 2))
        bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

        # 流入部、流出部の設定
        bc[0, :] = np.array([4.0, 0.0])
        bc_mask[0, :] = 2
        bc[-1, :] = np.array([4.0, 0.0])
        bc_mask[-1, :] = 2

        # 円柱ランダム生成
        ref_resolution = 500
        points = np.random.uniform(0, 2 * resolution, (100, 2))
        points = points[points[:, 1] < resolution]
        r = 16 * (resolution / ref_resolution)
        for p in points:
            l_ = np.round(np.maximum(p - r, 0)).astype(np.int32)
            u0 = round(min(p[0] + r, 2 * resolution))
            u1 = round(min(p[1] + r, resolution))
            for i in range(l_[0], u0):
                for j in range(l_[1], u1):
                    x = np.array([i, j]) + 0.5
                    if np.linalg.norm(x - p) < r:
                        bc[i, j] = np.array([0.0, 0.0])
                        bc_mask[i, j] = 1

        # 壁の設定
        bc[:, 0:2] = np.array([0.0, 0.0])
        bc_mask[:, 0:2] = 1
        bc[:, -2:] = np.array([0.0, 0.0])
        bc_mask[:, -2:] = 1

        return to_field(bc, bc_mask)

    @ti.kernel
    def calc(self, vc: ti.template()):
        for i, j in vc:
            if self.bc_mask[i, j] > 0:
                vc[i, j] = self._const_bc[i, j]
            if i == (2 * self._resolution - 1) and 2 <= j < self._resolution - 2:
                # 安定のために逆流しないようにする、かつ流速に上限をつける
                vc[i, j] = ti.Vector([ti.min(ti.max(vc[i - 1, j].x, 0.0), 10.0), vc[i - 1, j].y])
