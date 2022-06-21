import argparse
from pathlib import Path

import taichi as ti

from fluid_simulator import DyeFluidSimulator, FluidSimulator


def main():
    parser = argparse.ArgumentParser(description="Fluid Simulator")
    parser.add_argument(
        "-bc",
        "--boundary_condition",
        help="Boundary condition number",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
    )
    parser.add_argument("-re", "--reynolds_num", help="Reynolds number", type=float, default=10.0)
    parser.add_argument("-res", "--resolution", help="Resolution of y-axis", type=int, default=400)
    parser.add_argument("-dt", "--time_step", help="Time step", type=float, default=0.01)
    parser.add_argument(
        "-vis",
        "--visualization",
        help="Flow visualization type",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
    )
    parser.add_argument(
        "-vor_eps",
        "--vorticity_confinement_eps",
        help="Vorticity Confinement eps. 0.0 is disable.",
        type=float,
        default=4.0,
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

    n_bc = args.boundary_condition
    dt = args.time_step
    re = args.reynolds_num
    resolution = args.resolution
    vis_num = args.visualization
    no_dye = args.no_dye
    scheme = args.advection_scheme
    vor_eps = args.vorticity_confinement_eps if args.vorticity_confinement_eps != 0.0 else None

    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        device_memory_GB = 2.0 if resolution > 1000 else 1.0
        ti.init(arch=ti.gpu, device_memory_GB=device_memory_GB)

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    if no_dye:
        fluid_sim = FluidSimulator.create(n_bc, resolution, dt, re, vor_eps, scheme)
    else:
        fluid_sim = DyeFluidSimulator.create(n_bc, resolution, dt, re, vor_eps, scheme)

    img_path = Path(__file__).parent.resolve() / "result"

    # video_manager = ti.tools.VideoManager(output_dir=str(img_path), framerate=30, automatic_build=False)

    n_vis = 3 if no_dye else 4
    count = 0
    ss_count = 0
    paused = False
    while window.running:
        if not paused:
            fluid_sim.step()

        if vis_num == 0:
            img = fluid_sim.get_norm_field()
        elif vis_num == 1:
            img = fluid_sim.get_pressure_field()
        elif vis_num == 2:
            img = fluid_sim.get_vorticity_field()
        elif vis_num == 3:
            img = fluid_sim.get_dye_field()
        else:
            raise NotImplementedError()

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused
            elif e.key == "v":
                vis_num = (vis_num + 1) % n_vis
            elif e.key == "s":
                img_path.mkdir(exist_ok=True)
                ti.tools.imwrite(img, str(img_path / f"{ss_count:04}.png"))
                ss_count += 1

        canvas.set_image(img)
        window.show()

        # if count % 20 == 0:
        #     video_manager.write_frame(img)

        count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
