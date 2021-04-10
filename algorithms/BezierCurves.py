import numpy as np
import scipy.special
from bokeh.plotting import figure, show, output_file


def bezier(points, steps=100):
    n = len(points)
    b = [scipy.special.binom(n - 1, i) for i in range(n)]
    r = np.arange(n)

    for t in np.linspace(0, 1, steps):
        u = np.power(t, r) * np.power(1 - t, n - r - 1) * b
        yield t, u @ points


output_file("bezier.html")

points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]] * 4)
plot = figure()

for i in range(3, 14):
    curve = np.array([p for _, p in bezier(points[:i])])
    color = tuple(np.random.randint(0, 256, 3))
    plot.line(curve[:, 0], curve[:, 1], color=color)

show(plot)

