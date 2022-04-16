import numpy as np
import taichi as ti

np.random.seed(123)


def to_field(bc, bc_mask):
    bc_field = ti.Vector.field(2, ti.types.f64, shape=(bc.shape[0], bc.shape[1]))
    bc_field.from_numpy(bc)
    bc_mask_field = ti.field(ti.types.u8, shape=(bc_mask.shape[0], bc_mask.shape[1]))
    bc_mask_field.from_numpy(bc_mask)

    return bc_field, bc_mask_field


@ti.func
def set_const_bc(vc, bc_const, bc_mask, i, j):
    if bc_mask[i, j] > 0:
        vc[i, j] = bc_const[i, j]


@ti.func
def set_wall_bc(vc, pc, bc_mask, i, j):
    """壁の境界条件を設定する"""
    if bc_mask[i - 1, j] == 0 and bc_mask[i, j] == 1 and bc_mask[i + 1, j] == 1:
        vc[i, j].x == 0.0
        vc[i + 1, j].x = vc[i - 1, j].x
        pc[i, j] = pc[i - 1, j]
    elif bc_mask[i - 1, j] == 1 and bc_mask[i, j] == 1 and bc_mask[i + 1, j] == 0:
        vc[i, j].x == 0.0
        vc[i - 1, j].x = vc[i + 1, j].x
        pc[i, j] = pc[i + 1, j]
    elif bc_mask[i, j - 1] == 0 and bc_mask[i, j] == 1 and bc_mask[i, j + 1] == 1:
        vc[i, j].y == 0.0
        vc[i, j + 1].y = vc[i, j - 1].y
        pc[i, j] = pc[i, j - 1]
    elif bc_mask[i, j - 1] == 1 and bc_mask[i, j] == 1 and bc_mask[i, j + 1] == 0:
        vc[i, j].y == 0.0
        vc[i, j - 1].y = vc[i, j + 1].y
        pc[i, j] = pc[i, j + 1]


@ti.data_oriented
class BoundaryCondition1:
    def __init__(self, resolution):
        self.resolution = resolution
        self._bc_const, self._bc_mask = BoundaryCondition1._create_bc(resolution)

    @staticmethod
    def _create_bc(resolution):
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
    def calc(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            set_const_bc(vc, bc_const, bc_mask, i, j)
            set_wall_bc(vc, pc, bc_mask, i, j)
            if i == (2 * self.resolution - 1) and 0 <= j < self.resolution:
                vc[i, j] = ti.Vector([ti.min(ti.max(vc[i - 1, j].x, 0.0), 10.0), vc[i - 1, j].y])

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1


@ti.data_oriented
class BoundaryCondition2:
    def __init__(self, resolution):
        self.resolution = resolution
        self._bc_const, self._bc_mask = BoundaryCondition2._create_bc(resolution)

    @staticmethod
    def _create_bc(resolution):
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
        bc[0, 2 * y_point : 4 * y_point] = np.array([6.0, 0.0])
        bc_mask[0, 2 * y_point : 4 * y_point] = 2
        bc[-1, 2 * y_point : 4 * y_point] = np.array([6.0, 0.0])
        bc_mask[-1, 2 * y_point : 4 * y_point] = 2

        return to_field(bc, bc_mask)

    @ti.kernel
    def calc(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            set_const_bc(vc, bc_const, bc_mask, i, j)
            set_wall_bc(vc, pc, bc_mask, i, j)
            if (
                i == (2 * self.resolution - 1)
                and 2 * self.resolution // 6 <= j < 4 * self.resolution // 6
            ):
                vc[i, j] = ti.Vector([ti.min(ti.max(vc[i - 1, j].x, 0.0), 10.0), vc[i - 1, j].y])

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1


@ti.data_oriented
class BoundaryCondition3:
    def __init__(self, resolution):
        self.resolution = resolution
        self._bc_const, self._bc_mask = BoundaryCondition3._create_bc(resolution)

    @staticmethod
    def _create_bc(resolution):
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
    def calc(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            set_const_bc(vc, bc_const, bc_mask, i, j)
            set_wall_bc(vc, pc, bc_mask, i, j)
            if i == (2 * self.resolution - 1) and 2 <= j < self.resolution - 2:
                vc[i, j] = bc_const[i, j]

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1
