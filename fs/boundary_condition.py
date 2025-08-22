from pathlib import Path

import numpy as np
import numpy.typing as npt
import taichi as ti
from PIL import Image

from fs.differentiation import sample


@ti.data_oriented
class BoundaryCondition:
    def __init__(self, bc_const: npt.NDArray, bc_mask: npt.NDArray) -> None:
        self._bc_const, self._bc_mask = BoundaryCondition.to_field(bc_const, bc_mask)

    @ti.kernel
    def set_velocity_boundary_condition(self, vc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN201
        bc_mask = ti.static(self._bc_mask)
        for i, j in vc:
            if (
                bc_mask[i, j] == 1
                and 1 <= i < bc_mask.shape[0] - 1
                and 1 <= j < bc_mask.shape[1] - 1
            ):
                # for kk scheme
                # 壁内部の境界条件を設定、壁の厚さは片側2ピクセル以上を仮定する
                if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                    vc[i + 1, j] = -sample(vc, i - 1, j)
                elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                    vc[i - 1, j] = -sample(vc, i + 1, j)
                elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                    vc[i, j + 1] = -sample(vc, i, j - 1)
                elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                    vc[i, j - 1] = -sample(vc, i, j + 1)
            elif bc_mask[i, j] == 2:
                vc[i, j] = self._bc_const[i, j]
            elif bc_mask[i, j] == 3:
                # 逆流しないようにする
                vc[i, j].x = ti.max(sample(vc, i - 1, j).x, 0.05)  # type: ignore[reportAttributeAccessIssue]

    @ti.kernel
    def set_pressure_boundary_condition(self, pc: ti.template()):  # type: ignore[valid-type]  # noqa: ANN201
        bc_mask = ti.static(self._bc_mask)
        for i, j in pc:
            if bc_mask[i, j] == 1:
                if bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                    pc[i, j] = sample(pc, i - 1, j)
                elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 1 and bc_mask[i, j + 1] == 1:
                    pc[i, j] = sample(pc, i + 1, j)
                elif bc_mask[i, j - 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                    pc[i, j] = sample(pc, i, j - 1)
                elif bc_mask[i, j + 1] == 0 and bc_mask[i - 1, j] == 1 and bc_mask[i + 1, j] == 1:
                    pc[i, j] = sample(pc, i, j + 1)
                elif bc_mask[i - 1, j] == 0 and bc_mask[i, j + 1] == 0:
                    pc[i, j] = (sample(pc, i - 1, j) + sample(pc, i, j + 1)) / 2.0
                elif bc_mask[i + 1, j] == 0 and bc_mask[i, j + 1] == 0:
                    pc[i, j] = (sample(pc, i + 1, j) + sample(pc, i, j + 1)) / 2.0
                elif bc_mask[i - 1, j] == 0 and bc_mask[i, j - 1] == 0:
                    pc[i, j] = (sample(pc, i - 1, j) + sample(pc, i, j - 1)) / 2.0
                elif bc_mask[i + 1, j] == 0 and bc_mask[i, j - 1] == 0:
                    pc[i, j] = (sample(pc, i + 1, j) + sample(pc, i, j - 1)) / 2.0
            elif bc_mask[i, j] == 2:
                pc[i, j] = sample(pc, i + 1, j)
            elif bc_mask[i, j] == 3:
                pc[i, j] = 0.0

    @ti.func
    def is_wall(self, i: int, j: int) -> bool:
        return self._bc_mask[i, j] == 1

    @ti.func
    def is_fluid_domain(self, i: int, j: int) -> bool:
        return self._bc_mask[i, j] == 0

    def get_resolution(self) -> tuple[int, int]:
        return self._bc_const.shape[:2]

    @staticmethod
    def to_field(bc: npt.NDArray, bc_mask: npt.NDArray) -> tuple[ti.Field, ti.Field]:
        bc_field = ti.Vector.field(2, ti.f32, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_mask_field


class DyeBoundaryCondition(BoundaryCondition):
    def __init__(self, bc_const: npt.NDArray, bc_dye: npt.NDArray, bc_mask: npt.NDArray) -> None:
        self._bc_const, self._bc_dye, self._bc_mask = DyeBoundaryCondition.to_field(
            bc_const, bc_dye, bc_mask
        )

    @ti.kernel
    def set_dye_boundary_condition(self, dye: ti.template()):  # type: ignore[valid-type]  # noqa: ANN201
        bc_mask = ti.static(self._bc_mask)
        for i, j in dye:
            if bc_mask[i, j] == 2:
                dye[i, j] = self._bc_dye[i, j]

    @staticmethod
    def to_field(  # type: ignore[override]
        bc: npt.NDArray, bc_dye: npt.NDArray, bc_mask: npt.NDArray
    ) -> tuple[ti.Field, ti.Field, ti.Field]:
        bc_field = ti.Vector.field(2, ti.f32, shape=bc.shape[:2])
        bc_field.from_numpy(bc)
        bc_dye_field = ti.Vector.field(3, ti.f32, shape=bc_dye.shape[:2])
        bc_dye_field.from_numpy(bc_dye)
        bc_mask_field = ti.field(ti.u8, shape=bc_mask.shape[:2])
        bc_mask_field.from_numpy(bc_mask)

        return bc_field, bc_dye_field, bc_mask_field


def create_bc_array(
    x_resolution: int, y_resolution: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    bc = np.zeros((x_resolution, y_resolution, 2), dtype=np.float32)
    bc_mask = np.zeros((x_resolution, y_resolution), dtype=np.uint8)
    bc_dye = np.zeros((x_resolution, y_resolution, 3), dtype=np.float32)

    return bc, bc_mask, bc_dye


def create_color_map(color_list: list[npt.NDArray], n_samples: int) -> npt.NDArray:
    color_arr = np.vstack(color_list)
    x = np.linspace(0.0, 1.0, color_arr.shape[0], endpoint=True)

    x_ = np.linspace(0.0, 1.0, n_samples, endpoint=True)
    r_arr = np.interp(x_, x, color_arr[:, 0])
    g_arr = np.interp(x_, x, color_arr[:, 1])
    b_arr = np.interp(x_, x, color_arr[:, 2])

    return np.vstack((r_arr, g_arr, b_arr)).T


def set_circle(
    bc: npt.NDArray,
    bc_mask: npt.NDArray,
    bc_dye: npt.NDArray,
    center: tuple[int, int],
    radius: float,
) -> None:
    center_arr = np.asarray(center)
    l_ = np.round(np.maximum(center_arr - radius, 0)).astype(np.int32)
    u0 = round(min(center[0] + radius, bc.shape[0]))
    u1 = round(min(center[1] + radius, bc.shape[1]))
    for i in range(l_[0], u0):
        for j in range(l_[1], u1):
            x = np.array([i, j]) + 0.5
            if np.linalg.norm(x - center) < radius:
                bc[i, j] = np.array([0.0, 0.0])
                bc_mask[i, j] = 1
                bc_dye[i, j] = np.array([0.0, 0.0, 0.0])


def set_plane(
    bc: npt.NDArray,
    bc_mask: npt.NDArray,
    bc_dye: npt.NDArray,
    lower_left: tuple[int, int] | npt.NDArray,
    upper_right: tuple[int, int] | npt.NDArray,
) -> None:
    bc[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = np.array([0.0, 0.0])
    bc_mask[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = 1
    bc_dye[lower_left[0] : upper_right[0], lower_left[1] : upper_right[1]] = np.array(
        [0.0, 0.0, 0.0]
    )


def set_obstacle_fromfile(
    bc: npt.NDArray, bc_mask: npt.NDArray, bc_dye: npt.NDArray, filepath: Path
) -> None:
    """画像ファイルをもとに障害物を設定する.

    黒の領域を障害物として設定する
    """
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


def get_boundary_condition(
    num: int, resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    if num == 1:
        boundary_condition = create_boundary_condition1(resolution, enable_dye=enable_dye)
    elif num == 2:
        boundary_condition = create_boundary_condition2(resolution, enable_dye=enable_dye)
    elif num == 3:
        boundary_condition = create_boundary_condition3(resolution, enable_dye=enable_dye)
    elif num == 4:
        boundary_condition = create_boundary_condition4(resolution, enable_dye=enable_dye)
    elif num == 5:
        boundary_condition = create_boundary_condition5(resolution, enable_dye=enable_dye)
    elif num == 6:
        boundary_condition = create_boundary_condition6(resolution, enable_dye=enable_dye)
    else:
        raise NotImplementedError

    return boundary_condition


def create_boundary_condition1(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow() -> None:
        bc[:2, :] = np.array([1.0, 0.0])
        bc_mask[:2, :] = 2

        y = np.array([1.1, 1.1, 0.2])
        b = np.array([0.2, 0.2, 1.1])
        r = np.array([1.1, 0.2, 0.2])
        c = np.array([0.2, 1.1, 1.1])
        color_map = create_color_map([c, r, b, y] * 3, bc_dye.shape[1])
        bc_dye[:2, :] = np.stack((color_map, color_map), axis=0)

    # 流出部の設定
    def set_outflow() -> None:
        bc[-1, :] = np.array([0.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall() -> None:
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        # 円柱の設定
        r = y_res // 18
        c = (x_res // 4, y_res // 2)
        set_circle(bc, bc_mask, bc_dye, c, r)

    set_inflow()
    set_outflow()
    set_wall()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition2(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow() -> None:
        bc[:2, :] = np.array([1.0, 0.0])
        bc_mask[:2, :] = 2
        bc_dye[:2, :] = np.array([0.2, 0.2, 1.2])
        width = y_res // 10
        for i in range(0, y_res, width):
            bc_dye[:2, i : i + width // 2] = np.array([1.2, 1.2, 0.2])

    # 壁の設定
    def set_wall() -> None:
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
    def set_outflow() -> None:
        y_point = y_res // 3
        bc[-2:, y_point : 2 * y_point] = np.array([0.0, 0.0])
        bc_mask[-2:, y_point : 2 * y_point] = 3

    set_inflow()
    set_wall()
    set_outflow()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition3(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow() -> None:
        bc[:2, :] = np.array([1.0, 0.0])
        bc_mask[:2, :] = 2

        y = np.array([1.1, 1.1, 0.2])
        b = np.array([0.2, 0.2, 1.1])
        r = np.array([1.1, 0.2, 0.2])
        c = np.array([0.2, 1.1, 1.1])
        color_map = create_color_map([c, r, b, y], bc_dye.shape[1])
        bc_dye[:2, :] = np.stack((color_map, color_map), axis=0)

    # 流出部の設定
    def set_outflow() -> None:
        bc[-1, :] = np.array([0.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall() -> None:
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        # 円柱ランダム生成
        ref_resolution = 500
        np.random.seed(123)  # noqa: NPY002
        points = np.random.uniform(0, x_res, (100, 2))  # noqa: NPY002
        points = points[points[:, 1] < y_res]
        r = 16 * (y_res / ref_resolution)
        for p in points:
            set_circle(bc, bc_mask, bc_dye, p, r)

    set_inflow()
    set_outflow()
    set_wall()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition4(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 壁の設定
    def set_wall() -> None:
        set_plane(bc, bc_mask, bc_dye, (0, 0), (2, y_res))  # 左
        set_plane(bc, bc_mask, bc_dye, (x_res - 2, 0), (x_res, y_res))  # 右
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

    # 流入部の設定
    def set_inflow() -> None:
        y = np.array([1.1, 1.1, 0.2])
        b = np.array([0.2, 0.2, 1.1])
        r = np.array([1.1, 0.2, 0.2])
        c = np.array([0.2, 1.1, 1.1])
        color_map = create_color_map([c, r, b, y], y_res // 4 - 2)
        bc_dye[:2, 3 * y_res // 4 : -2] = np.stack((color_map, color_map), axis=0)
        bc_dye[:2, 2 : y_res // 4] = np.stack((color_map, color_map), axis=0)

        # 左上
        bc[:2, 3 * y_res // 4 : -2] = np.array([1.0, 0.0])
        bc_mask[:2, 3 * y_res // 4 : -2] = 2
        # 左下
        bc[:2, 2 : y_res // 4] = np.array([1.0, 0.0])
        bc_mask[:2, 2 : y_res // 4] = 2

    # 流出部の設定
    def set_outflow() -> None:
        # 中央
        bc[-2:, 3 * y_res // 8 : 5 * y_res // 8] = np.array([0.0, 0.0])
        bc_mask[-2:, 3 * y_res // 8 : 5 * y_res // 8] = 3

    set_wall()
    set_inflow()
    set_outflow()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition5(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流出部の設定
    def set_outflow() -> None:
        bc[-2:, :] = np.array([0.0, 0.0])
        bc_mask[-2:, :] = 3

    # 流入部の設定
    def set_inflow() -> None:
        bc[:2, 2 : y_res // 3] = np.array([1.0, 0.0])
        bc_mask[:2, 2 : y_res // 3] = 2
        bc_dye[:2, 2 : y_res // 3] = np.array([1.2, 0.2, 0.2])

        bc[:2, 2 * y_res // 3 : y_res - 2] = np.array([1.0, 0.0])
        bc_mask[:2, 2 * y_res // 3 : y_res - 2] = 2
        bc_dye[:2, 2 * y_res // 3 : y_res - 2] = np.array([0.2, 1.2, 1.2])

    # 壁の設定
    def set_wall() -> None:
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        size = x_res // 64
        set_plane(
            bc, bc_mask, bc_dye, (0, y_res // 5), (11 * x_res // 30, 4 * y_res // 5)
        )  # 左真中
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
        for a, b in zip(*params, strict=True):
            for i in range(1, 6 + b):
                p = np.array([a * x_res // 12, i * y_point - b * y_res // 12])
                lower_left = p - v
                upper_right = p + v
                set_plane(bc, bc_mask, bc_dye, lower_left, upper_right)

    set_inflow()
    set_outflow()
    set_wall()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition


def create_boundary_condition6(
    resolution: int, *, enable_dye: bool
) -> BoundaryCondition | DyeBoundaryCondition:
    # 1: 壁, 2: 流入部, 3: 流出部
    x_res, y_res = 2 * resolution, resolution
    bc, bc_mask, bc_dye = create_bc_array(x_res, y_res)

    # 流入部の設定
    def set_inflow() -> None:
        bc[:2, :] = np.array([1.0, 0.0])
        bc_mask[:2, :] = 2

        y = np.array([1.1, 1.1, 0.2])
        b = np.array([0.2, 0.2, 1.1])
        r = np.array([1.1, 0.2, 0.2])
        c = np.array([0.2, 1.1, 1.1])
        color_map = create_color_map([c, r, b, y], bc_dye.shape[1])
        bc_dye[:2, :] = np.stack((color_map, color_map), axis=0)

    # 流出部の設定
    def set_outflow() -> None:
        bc[-1, :] = np.array([0.0, 0.0])
        bc_mask[-1, :] = 3

    # 壁の設定
    def set_wall() -> None:
        set_plane(bc, bc_mask, bc_dye, (0, 0), (x_res, 2))  # 下
        set_plane(bc, bc_mask, bc_dye, (0, y_res - 2), (x_res, y_res))  # 上

        basepath = Path(__file__).parents[1].resolve()
        filepath = basepath / "images/bc_mask/dragon.png"
        set_obstacle_fromfile(bc, bc_mask, bc_dye, filepath)

    set_inflow()
    set_outflow()
    set_wall()

    if not enable_dye:
        boundary_condition = BoundaryCondition(bc, bc_mask)
    else:
        boundary_condition = DyeBoundaryCondition(bc, bc_dye, bc_mask)

    return boundary_condition
