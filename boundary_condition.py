from pathlib import Path
import numpy as np
import taichi as ti


@ti.data_oriented
class BoundaryCondition:
    def __init__(self, bc_const, bc_mask):
        self._bc_const, self._bc_mask = BoundaryCondition.to_field(bc_const, bc_mask)

    @ti.kernel
    def set_boundary_condition(self, vc: ti.template(), pc: ti.template()):
        bc_const, bc_mask = ti.static(self._bc_const, self._bc_mask)
        for i, j in vc:
            self._set_wall_bc(vc, bc_const, bc_mask, i, j)
            self._set_inflow_bc(vc, bc_const, bc_mask, i, j)
            self._set_outflow_bc(vc, pc, bc_mask, i, j)
            if 0 < i < vc.shape[0] - 1 and 0 < j < vc.shape[1] - 1:
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
        # 右側のみ
        if bc_mask[i, j] == 3:
            vc[i, j].x = min(max(vc[i - 1, j].x, 0.0), 10.0)  # 逆流しないようにする
            vc[i, j].y = min(max(vc[i - 1, j].y, -10.0), 10.0)
            pc[i, j] = 0.0

    @ti.func
    def _set_inside_wall_bc(self, vc, pc, bc_mask, i, j):
        """壁内部の境界条件を設定する

        ※ 壁の厚さは片側2ピクセル以上を仮定
        """
        if bc_mask[i, j] == 1:
            if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                vc[i, j] = [0.0, 0.0]
                pc[i, j] = pc[i - 1, j]
                vc[i + 1, j] = -vc[i - 1, j]  # for kk scheme
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                vc[i, j] = [0.0, 0.0]
                pc[i, j] = pc[i + 1, j]
                vc[i - 1, j] = -vc[i + 1, j]  # for kk scheme
            elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                vc[i, j] = [0.0, 0.0]
                pc[i, j] = pc[i, j - 1]
                vc[i, j + 1] = -vc[i, j - 1]  # for kk scheme
            elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                vc[i, j] = [0.0, 0.0]
                pc[i, j] = pc[i, j + 1]
                vc[i, j - 1] = -vc[i, j + 1]  # for kk scheme
            elif bc_mask[i - 1, j] == 0 and bc_mask[i, j + 1] == 0:
                vc[i, j] = -(vc[i - 1, j] + vc[i, j + 1]) / 2.0
                pc[i, j] = (pc[i - 1, j] + pc[i, j + 1]) / 2.0
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j + 1] == 0:
                vc[i, j] = -(vc[i + 1, j] + vc[i, j + 1]) / 2.0
                pc[i, j] = (pc[i + 1, j] + pc[i, j + 1]) / 2.0
            elif bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 0:
                vc[i, j] = -(vc[i - 1, j] + vc[i, j - 1]) / 2.0
                pc[i, j] = (pc[i - 1, j] + pc[i, j - 1]) / 2.0
            elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 0:
                vc[i, j] = -(vc[i + 1, j] + vc[i, j - 1]) / 2.0
                pc[i, j] = (pc[i + 1, j] + pc[i, j - 1]) / 2.0

    @staticmethod
    def to_field(bc, bc_mask):
        bc_field = ti.Vector.field(2, ti.f32, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_mask_field


class DyeBoundaryCondition(BoundaryCondition):
    def __init__(self, bc_const, bc_dye, bc_mask):
        self._bc_const, self._bc_dye, self._bc_mask = DyeBoundaryCondition.to_field(
            bc_const, bc_dye, bc_mask
        )

    @ti.kernel
    def set_boundary_condition(self, vc: ti.template(), pc: ti.template(), dye: ti.template()):
        bc_const, bc_dye, bc_mask = ti.static(self._bc_const, self._bc_dye, self._bc_mask)
        for i, j in vc:
            self._set_wall_bc(vc, bc_const, bc_mask, i, j)
            self._set_inflow_bc(vc, bc_const, bc_mask, i, j)
            self._set_indye_bc(dye, bc_dye, bc_mask, i, j)
            self._set_outflow_bc(vc, pc, bc_mask, i, j)
            if 0 < i < vc.shape[0] - 1 and 0 < j < vc.shape[1] - 1:
                self._set_inside_wall_bc(vc, pc, bc_mask, i, j)

    @staticmethod
    def to_field(bc, bc_dye, bc_mask):
        bc_field = ti.Vector.field(2, ti.f32, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        bc_dye_field = ti.Vector.field(3, ti.f32, shape=bc_dye.shape[:2])
        bc_dye_field.from_numpy(bc_dye)
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_dye_field, bc_mask_field

    @ti.func
    def _set_indye_bc(self, dye, bc_dye, bc_mask, i, j):
        if bc_mask[i, j] == 2:
            dye[i, j] = bc_dye[i, j]


def create_bc_array(x_resolution, y_resolution):
    bc = np.zeros((x_resolution, y_resolution, 2), dtype=np.float32)
    bc_mask = np.zeros((x_resolution, y_resolution), dtype=np.uint8)
    bc_dye = np.zeros((x_resolution, y_resolution, 3), dtype=np.float32)

    return bc, bc_mask, bc_dye


def set_circle(bc, bc_mask, bc_dye, center, radius):
    center = np.asarray(center)
    l_ = np.round(np.maximum(center - radius, 0)).astype(np.int32)
    u0 = round(min(center[0] + radius, bc.shape[0]))
    u1 = round(min(center[1] + radius, bc.shape[1]))
    for i in range(l_[0], u0):
        for j in range(l_[1], u1):
            x = np.array([i, j]) + 0.5
            if np.linalg.norm(x - center) < radius:
                bc[i, j] = np.array([0.0, 0.0])
                bc_mask[i, j] = 1
                bc_dye[i, j] = np.array([0.0, 0.0, 0.0])


def set_plane(bc, bc_mask, bc_dye, lower_left, upper_right):
    bc[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = np.array([0.0, 0.0])
    bc_mask[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = 1
    bc_dye[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = np.array(
        [0.0, 0.0, 0.0]
    )


def set_obstacle_fromfile(bc, bc_mask, bc_dye, filepath):
    """画像ファイルをもとに障害物を設定する

    RGB値がBlackの領域を障害物として設定する
    """
    from PIL import Image

    image = Image.open(filepath).convert("L")
    x_res, y_res = bc.shape[:2]

    # アスペクト比を維持して画像サイズをy_resに合わせてリサイズする
    x_ratio = x_res / image.width
    y_ratio = y_res / image.height
    resize_size = (
        (x_res, round(image.height * x_ratio))
        if x_ratio < y_ratio
        else (round(image.width * y_ratio), y_res)
    )
    image = image.resize(resize_size)

    # 領域の中央へ移動する
    image_ = Image.new(image.mode, (x_res, y_res), 255)
    image_.paste(image, ((x_res - image.width) // 2, 0))

    mask_indices = np.flip(np.array(image_).T, axis=1) < 200
    bc[mask_indices] = np.array([0.0, 0.0])
    bc_mask[mask_indices] = 1
    bc_dye[mask_indices] = np.array([0.0, 0.0, 0.0])


def create_boundary_condition1(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow():
        bc[:2, :] = np.array([20.0, 0.0])
        bc_dye[:2, :] = np.array([0.2, 0.2, 1.2])
        width = y_res // 10
        for i in range(0, y_res, width):
            bc_dye[:2, i : i + width // 2] = np.array([1.2, 1.2, 0.2])
        bc_mask[:2, :] = 2

    # 流出部の設定
    def set_outflow():
        bc[-1, :] = np.array([20.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        # 円柱の設定
        r = y_res // 18
        c = (x_res // 4, y_res // 2)
        set_circle(bc, bc_mask, bc_dye, c, r)

    set_inflow()
    set_outflow()
    set_wall()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition2(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow():
        bc[:2, :] = np.array([20.0, 0.0])
        bc_mask[:2, :] = 2
        bc_dye[:2, :] = np.array([0.2, 0.2, 1.2])
        width = y_res // 10
        for i in range(0, y_res, width):
            bc_dye[:2, i : i + width // 2] = np.array([1.2, 1.2, 0.2])

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (2, y_res // 3))  # 左下
        set_plane(bc, bc_mask, bc_dye, (0, 2 * y_res // 3), (2, y_res))  # 左上
        set_plane(bc, bc_mask, bc_dye, (x_res - 2, 0), (x_res, y_res))  # 右
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        x_point = x_res // 5
        y_point = y_res // 2
        size = y_res // 32  # 壁幅
        # 左
        set_plane(bc, bc_mask, bc_dye, (x_point - size, y_point), (x_point + size, y_res))
        # 真ん中左
        set_plane(bc, bc_mask, bc_dye, (2 * x_point - size, 0), (2 * x_point + size, y_point))
        # 真ん中右
        set_plane(bc, bc_mask, bc_dye, (3 * x_point - size, y_point), (3 * x_point + size, y_res))
        # 右
        set_plane(bc, bc_mask, bc_dye, (4 * x_point - size, 0), (4 * x_point + size, y_point))

    # 流出部の設定
    def set_outflow():
        y_point = y_res // 6
        bc[-2:, 2 * y_point : 4 * y_point] = np.array([15.0, 0.0])
        bc_mask[-2:, 2 * y_point : 4 * y_point] = 3

    set_inflow()
    set_wall()
    set_outflow()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition3(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow():
        bc[:2, :] = np.array([20.0, 0.0])
        bc_dye[:2, :] = np.array([0.2, 0.2, 1.2])
        width = y_res // 10
        for i in range(0, y_res, width):
            bc_dye[:2, i : i + width // 2] = np.array([1.2, 1.2, 0.2])
        bc_mask[:2, :] = 2

    # 流出部の設定
    def set_outflow():
        bc[-1, :] = np.array([15.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        # 円柱ランダム生成
        ref_resolution = 500
        np.random.seed(123)
        points = np.random.uniform(0, x_res, (100, 2))
        points = points[points[:, 1] < y_res]
        r = 16 * (y_res / ref_resolution)
        for p in points:
            set_circle(bc, bc_mask, bc_dye, p, r)

    set_inflow()
    set_outflow()
    set_wall()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition4(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (2, y_res))  # 左
        set_plane(bc, bc_mask, bc_dye, (x_res - 2, 0), (x_res, y_res))  # 右
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

    # 流入部の設定
    def set_inflow():
        size = y_res // 5

        # 下
        bc[size : 2 * size, :2] = np.array([8.0, 16.0])
        bc_dye[size : 2 * size, :2] = np.array([1.2, 1.2, 0.2])
        bc_mask[size : 2 * size, :2] = 2

        # 上
        bc[-2 * size : -size, -2:] = np.array([-8.0, -16.0])
        bc_dye[-2 * size : -size, -2:] = np.array([0.2, 0.2, 1.2])
        bc_mask[-2 * size : -size, -2:] = 2

    set_wall()
    set_inflow()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition5(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流出部の設定
    def set_outflow():
        bc[-2:, :] = np.array([15.0, 0.0])
        bc_mask[-2:, :] = 3

    # 流入部の設定
    def set_inflow():
        #
        bc[:2, 2 : y_res // 3] = np.array([20.0, 0.0])
        bc_mask[:2, 2 : y_res // 3] = 2
        bc_dye[:2, 2 : y_res // 3] = np.array([1.2, 0.2, 0.2])

        bc[:2, 2 * y_res // 3 : y_res - 2] = np.array([20.0, 0.0])
        bc_mask[:2, 2 * y_res // 3 : y_res - 2] = 2
        bc_dye[:2, 2 * y_res // 3 : y_res - 2] = np.array([0.2, 1.2, 1.2])

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        size = x_res // 64
        set_plane(bc, bc_mask, bc_dye, (0, y_res // 5), (11 * x_res // 30, 4 * y_res // 5))  # 左真中
        set_plane(
            bc, bc_mask, bc_dye, (x_res // 2 - size, 0), (x_res // 2 + size, 2 * y_res // 5)
        )  # 下真中
        set_plane(
            bc, bc_mask, bc_dye, (x_res // 2 - size, 3 * y_res // 5), (x_res // 2 + size, y_res)
        )  # 上真中

        # 障害物
        y_point = y_res // 6
        v = np.array([y_res, y_res]) // 25
        params = [(7, 8, 9, 10, 11), (0, 1, 0, 1, 0)]
        for a, b in zip(*params):
            for i in range(1, 6 + b):
                p = np.array([a * x_res // 12, i * y_point - b * y_res // 12])
                lower_left = p - v
                upper_right = p + v
                set_plane(bc, bc_mask, bc_dye, lower_left, upper_right)

    set_outflow()
    set_inflow()
    set_wall()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition6(resolution, no_dye=False):
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow():
        bc[:2, :] = np.array([20.0, 0.0])
        bc_dye[:2, :] = np.array([0.2, 0.2, 1.2])
        width = y_res // 10
        for i in range(0, y_res, width):
            bc_dye[:2, i : i + width // 2] = np.array([1.2, 1.2, 0.2])
        bc_mask[:2, :] = 2

    # 流出部の設定
    def set_outflow():
        bc[-1, :] = np.array([15.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall():
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        basepath = Path(__file__).resolve().parent
        filepath = basepath / "images/bc_mask/dragon.png"
        set_obstacle_fromfile(bc, bc_mask, bc_dye, filepath)

    set_inflow()
    set_outflow()
    set_wall()

    if no_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition
