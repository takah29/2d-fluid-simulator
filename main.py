import taichi as ti
from fluid_simulator import FluidSimulator, DyeFluidSimulator


def main():
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    resolution = 800
    paused = False

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    dt = 0.02
    re = 1e8
    fluid_sim = DyeFluidSimulator.create(2, resolution, dt, re)
    n_vis = 4

    video_manager = ti.tools.VideoManager(output_dir="result", framerate=30, automatic_build=False)

    count = 0
    vis_num = 0
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
