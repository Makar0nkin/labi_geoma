import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sidrpy.classes import Point, Vector
from sidrpy.polygon_utils import convex_hull_jarvis, calc_diameter


fig, ax = plt.subplots(1, 1)
fps = 30

n = 10
x, y = np.random.rand(2, n)
points = [Point(*coord) for coord in zip(x, y)]

speed_const = 0.005
speed_vectors = list()
for _ in range(n):
    random_angle = np.random.rand() * 2 * np.pi  # np.pi
    speed_vectors.append(Vector(speed_const * np.cos(random_angle),
                                speed_const * np.sin(random_angle)))

convex_hull = convex_hull_jarvis([point.x for point in points],
                                 [point.y for point in points])
convex_hull = [*convex_hull, convex_hull[0]]
diam_points, d = calc_diameter(convex_hull).values()

fps = 60
frames_n = 250
d_max = d + 0.15
def new_frame(i):
    print(f'{i}/{frames_n}')
    convex_hull = convex_hull_jarvis([point.x for point in points],
                                     [point.y for point in points])
    convex_hull = [*convex_hull, convex_hull[0]]
    diam_points, d = calc_diameter(convex_hull).values()

    ax.clear()
    ax.scatter([coord.x for coord in points], [coord.y for coord in points], color="tab:blue")
    ax.plot([coord.x for coord in convex_hull], [coord.y for coord in convex_hull], color="tab:blue")
    ax.scatter([dp.x_arr for dp in diam_points], [dp.y for dp in diam_points], color="tab:orange")
    ax.plot([dp.x_arr for dp in diam_points], [dp.y for dp in diam_points], color="tab:orange")
    ax.set_xlim([-0.6, 1.6])
    ax.set_ylim([-0.6, 1.6])

    for ind in range(n):
        if points[ind] in diam_points and d > d_max:
            speed_vectors[ind] = -speed_vectors[ind]
        points[ind] += speed_vectors[ind]

anime = FuncAnimation(fig, new_frame, frames=frames_n + fps // 10,
                      interval=50, repeat=False)
anime.save(f"simple_animation_{fps}fps.mp4", dpi=300, writer=FFMpegWriter(fps=fps))