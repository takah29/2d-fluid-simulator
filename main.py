import argparse

import taichi as ti

from fluid_simulator import DyeFluidSimulator, FluidSimulator


def main():
    parser = argparse.ArgumentParser(description="Fluid Simulator")
    parser.add_argument(
        "-bc",
        "--boundary_condition",
        help="Boundary condition number",
        type=int,
        choices=[1, 2, 3, 4],
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

    args = parser.parse_args()

    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    n_bc = args.boundary_condition
    dt = args.time_step
    re = args.reynolds_num
    resolution = args.resolution
    vis_num = args.visualization
    no_dye = args.no_dye
    scheme = args.advection_scheme
    vor_eps = args.vorticity_confinement_eps if args.vorticity_confinement_eps != 0.0 else None

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    if no_dye:
        fluid_sim = FluidSimulator.create(n_bc, resolution, dt, re, vor_eps, scheme)
    else:
        fluid_sim = DyeFluidSimulator.create(n_bc, resolution, dt, re, vor_eps, scheme)

    video_manager = ti.tools.VideoManager(output_dir="result", framerate=30, automatic_build=False)

    n_vis = 3 if no_dye else 4
    count = 0
    paused = False
    img = None
    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused
            elif e.key == "v":
                vis_num = (vis_num + 1) % n_vis
            elif e.key == "s":
                video_manager.write_frame(img)

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

        if count % 20 == 0:
            canvas.set_image(img)
            window.show()

        # if count % 100 == 0:
        #     video_manager.write_frame(img)

        count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
