from math import pi, e
import taichi as ti


@ti.func
def visualize_norm(vec):
    c = ti.sqrt(vec.dot(vec))
    return ti.Vector([c, c, c])


@ti.func
def visualize_hue(vec):
    h = ti.atan2(vec.y, vec.x)

    while h < 0:
        h += 2 * pi
    h /= 2 * pi

    m = ti.sqrt(vec.x**2 + vec.y**2)
    ranges = 0
    rangee = 1

    while m > rangee:
        ranges = rangee
        rangee *= e

    k = (m - ranges) / (rangee - ranges)
    s = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
    s = 1 - pow(1 - s, 3)
    s = 0.4 + s * 0.6

    v = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
    v = 1 - v
    v = 1 - pow(1 - v, 3)
    v = 0.6 + v * 0.4

    return _hsv_to_rgb(h, s, v)

@ti.func
def visualize_xy(vec):
    return ti.Vector([vec.y, 0.0, vec.x])


@ti.func
def _hsv_to_rgb(h: float, s: float, v: float):

    if h == 1:
        h = 0
    z = ti.floor(h * 6)
    i = int(z)
    f = float(h * 6 - z)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    r = g = b = 1.0

    if i == 0:
        r = v
        g = t
        b = p
    elif i == 1:
        r = q
        g = v
        b = p
    elif i == 2:
        r = p
        g = v
        b = t
    elif i == 3:
        r = p
        g = q
        b = v
    elif i == 4:
        r = t
        g = p
        b = v
    elif i == 5:
        r = v
        g = p
        b = q

    return ti.Vector([r, g, b])
