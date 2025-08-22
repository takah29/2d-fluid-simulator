import argparse
from pathlib import Path

import numpy as np
import taichi as ti

from fs.fluid_simulator import DyeFluidSimulator, FluidSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluid Simulator")
    parser.add_argument(
        "-bc",
        "--boundary_condition",
        help="Boundary condition number",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
    )
    parser.add_argument(
        "-re", "--reynolds_num", help="Reynolds number", type=float, default=1000000.0
    )
    parser.add_argument("-res", "--resolution", help="Resolution of y-axis", type=int, default=400)
    parser.add_argument("-dt", "--time_step", help="Time step", type=float, default=0.0)
    parser.add_argument(
        "-vis",
        "--visualization",
        help="Flow visualization type",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
    )
    parser.add_argument(
        "-vc",
        "--vorticity_confinement",
        help="Vorticity Confinement. 0.0 is disable.",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "-scheme",
        "--advection_scheme",
        help="Advection Scheme",
        type=str,
        choices=["upwind", "kk", "cip"],
        default="cip",
    )
    parser.add_argument("-no_dye", "--no_dye", help="No dye calculation", action="store_true")
    parser.add_argument("-cpu", "--cpu", action="store_true")

    args = parser.parse_args()

    n_bc: int = args.boundary_condition
    re: float = args.reynolds_num
    resolution: int = args.resolution
    dt: float = args.time_step if args.time_step != 0.0 else 0.05 / resolution
    vis_num: int = args.visualization
    enable_dye: bool = not args.no_dye
    scheme: str = args.advection_scheme
    vor_eps: float | None = (
        args.vorticity_confinement if args.vorticity_confinement != 0.0 else None
    )
    dx: float = 1 / resolution

    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        device_memory_gb = 2.0 if resolution > 1000 else 1.0
        ti.init(arch=ti.gpu, device_memory_GB=device_memory_gb)

    print(
        f"Boundary Condition: {n_bc}\ndt: {dt}\nRe: {re}\nResolution: {resolution}\n"
        f"Scheme: {scheme}\nVorticity confinement: {vor_eps}"
    )

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    if enable_dye:
        fluid_sim = DyeFluidSimulator.create(n_bc, resolution, dt, dx, re, vor_eps, scheme)
    else:
        fluid_sim = FluidSimulator.create(n_bc, resolution, dt, dx, re, vor_eps, scheme)

    output_path = Path(__file__).parent.resolve() / "output"

    # video_manager = ti.tools.VideoManager(output_dir=str(img_path), framerate=30, automatic_build=False)

    n_vis = 4 if enable_dye else 3
    step = 0
    ss_count = 0
    paused = False
    img = None
    while window.running:
        if step % 5 == 0:
            if vis_num == 0:
                img = fluid_sim.get_norm_field()
            elif vis_num == 1:
                img = fluid_sim.get_pressure_field()
            elif vis_num == 2:
                img = fluid_sim.get_vorticity_field()
            elif vis_num == 3:
                img = fluid_sim.get_dye_field()  # type: ignore[attr-defined]
            else:
                raise NotImplementedError

            canvas.set_image(img)
            window.show()

            # video_manager.write_frame(img)

        if not paused:
            fluid_sim.step()

        if window.get_event(ti.ui.PRESS):
            e = window.event

            if e.key == ti.ui.ESCAPE:
                break

            if e.key == "p":
                paused = not paused
            elif e.key == "v":
                vis_num = (vis_num + 1) % n_vis
            elif e.key == "s":
                output_path.mkdir(exist_ok=True)
                if img is not None:
                    ti.tools.imwrite(img, str(output_path / f"{ss_count:04}.png"))
                ss_count += 1
            elif e.key == "d":
                output_path.mkdir(exist_ok=True)
                fields = fluid_sim.field_to_numpy()
                np.savez(str(output_path / f"step_{step:06}.npz"), **fields)

        step += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
