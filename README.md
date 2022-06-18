# 2D Fluid Simulator

![baundary_condition_2_dye](./images/bc2_res1600_re100_cip_vc4_dye.jpg)
![baundary_condition_2_norm](./images/bc2_res1600_re100_cip_vc4_norm.jpg)

## Features

- Finite Difference Method (MAC Method)
- Advection Scheme
  - Upwind Differencing
  - Kawamura-Kuwahara
  - CIP (Constrained Interpolation Profile)
- Flow Visualization
  - Norm (Velocity) and Pressure
  - Pressure
  - Vorticity
  - Dye
- Vorticity Confinement

## Requirements

- Python 3.9
- Taichi 1.0

GeForce GTX 1080 or higher recommended.

## Usage

- Boundary Condition 1, Reynolds Number = 0.4, dt = 0.01
  ```bash
  python main.py -re 0.4
  ```
  Press `V` key switches the flow visualization method.
- Boundary Condition 2, Reynolds Number = 100.0, resolution = 800, dt = 0.01
  ```bash
  python main.py -bc 2 -re 100 -res 800
  ```
- Boundary Condition 3, Reynolds Number = 100.0, resolution = 800, dt = 0.01, no vorticity confinement, upwind scheme
  ```bash
  python main.py -bc 3 -re 100 -res 800 -vor_eps 0.0 -scheme upwind
  ```
- Boundary Condition 3, Reynolds Number = 100.0, resolution = 800, dt = 0.01, no vorticity confinement, cip scheme
  ```bash
  python main.py -bc 3 -re 100 -res 800 -vor_eps 0.0
  ```
- Boundary Condition 5, Reynolds Number = 100.0, dt = 0.01
  ```bash
  python main.py -bc 5 -re 100
  ```
- Help
  ```bash
  python main.py -h
  ```

### For CPU

- Boundary Condition 2, Reynolds Number = 100.0, resolution = 200, dt = 0.01
  ```bash
  python main.py -bc 2 -re 100.0 -res 200 -cpu
  ```

## Screenshots

### Flow Visualization

- Norm and Pressure
  ![norm_and_pressure](./images/bc5_res800_re100_cip_vc4_norm.jpg)
- Pressure
  ![pressure](./images/bc5_res800_re100_cip_vc4_pressure.jpg)
- Vorticity
  ![vorticity](./images/bc5_res800_re100_cip_vc4_vorticity.jpg)
- Dye
  ![dye](./images/bc5_res800_re100_cip_vc4_dye.jpg)

### Vorticity Confinement

- eps = 0.0
  ![no_vorticity_confinement](./images/bc3_res800_re1000_cip_vc0_dye.jpg)
- eps = 4.0
  ![vorticity_confinement](./images/bc3_res800_re1000_cip_vc4_dye.jpg)

## References

- [移流法](http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?%B0%DC%CE%AE%CB%A1)
- [2 次元 CIP 法による移流項の計算](https://i-ric.org/yasu/nbook2/04_Chapt04.html#cip)
- [GPU Gems Chapter 38. Fast Fluid Dynamics Simulation on the GPU
  ](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [Ronald Fedkiw, Jos Stam, Henrik Wann Jensen. Visual Simulation of Smoke.](https://web.stanford.edu/class/cs237d/smoke.pdf)
