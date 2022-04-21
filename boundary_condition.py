import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition:
    def __init__(self, bc_const, bc_mask):
        self._bc_const, self._bc_mask = BoundaryCondition._to_field(bc_const, bc_mask)

    @ti.kernel
    def calc(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            self._set_wall_bc(vc, bc_const, bc_mask, i, j)
            self._set_inflow_bc(vc, bc_const, bc_mask, i, j)
            self._set_outflow_bc(vc, pc, bc_mask, i, j)
            if 0 < i < vc.shape[0] and 0 < j < vc.shape[1]:
                self._set_inside_wall_bc(vc, pc, bc_mask, i, j)

    @ti.func
    def is_wall(self, i, j):
        return self._bc_mask[i, j] == 1

    def get_resolution(self):
        return self._bc_const.shape[:2]

    @ti.func
    def _set_wall_bc(self, vc, bc_const, bc_mask, i, j):
        if bc_mask[i, j] == 1:
            vc[i, j] = bc_const[i, j]

    @ti.func
    def _set_inflow_bc(self, vc, bc_const, bc_mask, i, j):
        if bc_mask[i, j] == 2:
            vc[i, j] = bc_const[i, j]

    @ti.func
    def _set_outflow_bc(self, vc, pc, bc_mask, i, j):
        if bc_mask[i, j] == 3:
            vc[i, j].x = min(max(vc[i - 1, j].x, 0.0), 10.0)  # 逆流しないようにする
            vc[i, j].y = vc[i - 1, j].y
            pc[i, j] = 0.0

    @ti.func
    def _set_inside_wall_bc(self, vc, pc, bc_mask, i, j):
        """壁内部の境界条件を設定する"""
        if bc_mask[i - 1, j] == 0 and bc_mask[i, j] == 1 and bc_mask[i + 1, j] == 1:
            vc[i + 1, j].x = vc[i - 1, j].x
            pc[i, j] = pc[i - 1, j]
        elif bc_mask[i - 1, j] == 1 and bc_mask[i, j] == 1 and bc_mask[i + 1, j] == 0:
            vc[i - 1, j].x = vc[i + 1, j].x
            pc[i, j] = pc[i + 1, j]
        elif bc_mask[i, j - 1] == 0 and bc_mask[i, j] == 1 and bc_mask[i, j + 1] == 1:
            vc[i, j + 1].y = vc[i, j - 1].y
            pc[i, j] = pc[i, j - 1]
        elif bc_mask[i, j - 1] == 1 and bc_mask[i, j] == 1 and bc_mask[i, j + 1] == 0:
            vc[i, j - 1].y = vc[i, j + 1].y
            pc[i, j] = pc[i, j + 1]

    @staticmethod
    def _to_field(bc, bc_mask):
        bc_field = ti.Vector.field(2, ti.types.f64, shape=(bc.shape[0], bc.shape[1]))
        bc_field.from_numpy(bc)
        bc_mask_field = ti.field(ti.types.u8, shape=(bc_mask.shape[0], bc_mask.shape[1]))
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_mask_field

    @staticmethod
    def _set_circle(bc, bc_mask, i, j, radius):
        p = np.array([i, j])
        l_ = np.round(np.maximum(p - radius, 0)).astype(np.int32)
        u0 = round(min(p[0] + radius, bc.shape[0]))
        u1 = round(min(p[1] + radius, bc.shape[1]))
        for i in range(l_[0], u0):
            for j in range(l_[1], u1):
                x = np.array([i, j]) + 0.5
                if np.linalg.norm(x - p) < radius:
                    bc[i, j] = np.array([0.0, 0.0])
                    bc_mask[i, j] = 1


def create_boundary_condition1(resolution):
    # 1: 壁, 2: 流入部, 3: 流出部
    bc = np.zeros((2 * resolution, resolution, 2))
    bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

    # 流入部の設定
    bc[0, :] = np.array([10.0, 0.0])
    bc_mask[0, :] = 2
    # bc[0, resolution // 2 - 2 * size : resolution // 2 + 2 * size] = np.array([8.0, 0.0])
    # bc_mask[0, resolution // 2 - 2 * size : resolution // 2 + 2 * size] = 2

    # 流出部の設定
    bc[-1, :] = np.array([10.0, 0.0])
    bc_mask[-1, :] = 3

    # 壁の設定
    bc[:, :2] = np.array([0.0, 0.0])
    bc_mask[:, :2] = 1
    bc[:, -2:] = np.array([0.0, 0.0])
    bc_mask[:, -2:] = 1

    # 円柱の設定
    r = resolution // 18
    BoundaryCondition._set_circle(bc, bc_mask, resolution // 2 - r, resolution // 2, r)

    return BoundaryCondition(bc, bc_mask)


def create_boundary_condition2(resolution):
    bc = np.zeros((2 * resolution, resolution, 2))
    bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

    # 壁の設定
    size = resolution // 32  # 壁幅
    bc[:2, :] = np.array([0.0, 0.0])
    bc_mask[:2, :] = 1
    bc[-2:, :] = np.array([0.0, 0.0])
    bc_mask[-2:, :] = 1
    bc[:, :2] = np.array([0.0, 0.0])
    bc_mask[:, :2] = 1
    bc[:, -2:] = np.array([0.0, 0.0])
    bc_mask[:, -2:] = 1

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

    # 流入部の設定
    y_point = resolution // 6
    bc[:2, 2 * y_point : 4 * y_point] = np.array([6.0, 0.0])
    bc_mask[:2, 2 * y_point : 4 * y_point] = 2

    # 流出部の設定
    bc[-2:, 2 * y_point : 4 * y_point] = np.array([6.0, 0.0])
    bc_mask[-2:, 2 * y_point : 4 * y_point] = 3

    return BoundaryCondition(bc, bc_mask)


def create_boundary_condition3(resolution):
    bc = np.zeros((2 * resolution, resolution, 2))
    bc_mask = np.zeros((2 * resolution, resolution), dtype=np.uint8)

    # 流入部の設定
    bc[0, :] = np.array([8.0, 0.0])
    bc_mask[0, :] = 2

    # 流出部の設定
    bc[-1, :] = np.array([8.0, 0.0])
    bc_mask[-1, :] = 3

    # 壁の設定
    bc[:, :2] = np.array([0.0, 0.0])
    bc_mask[:, :2] = 1
    bc[:, -2:] = np.array([0.0, 0.0])
    bc_mask[:, -2:] = 1

    # 円柱ランダム生成
    ref_resolution = 500
    np.random.seed(123)
    points = np.random.uniform(0, 2 * resolution, (100, 2))
    points = points[points[:, 1] < resolution]
    r = 16 * (resolution / ref_resolution)
    for p in points:
        BoundaryCondition._set_circle(bc, bc_mask, p[0], p[1], r)

    return BoundaryCondition(bc, bc_mask)
