import taichi as ti
from fluid_simulator import FluidSimulator, DyesFluidSimulator


def main():
    arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
    ti.init(arch=arch)

    resolution = 400
    paused = False

    window = ti.ui.Window("Fluid Simulation", (2 * resolution, resolution), vsync=False)
    canvas = window.get_canvas()

    dt = 0.01
    re = 1e32
    fluid_sim = DyesFluidSimulator.create(2, resolution, dt, re)

    # video_manager = ti.tools.VideoManager(output_dir="result", framerate=60, automatic_build=False)

    count = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "p":
                paused = not paused

        if not paused:
            fluid_sim.step()

        img = fluid_sim.get_norm_field()

        if count % 10 == 0:
            canvas.set_image(img)
            window.show()

        # if count % 50 == 0:
        #     video_manager.write_frame(img)

        count += 1

    # video_manager.make_video(mp4=True)


if __name__ == "__main__":
    main()
