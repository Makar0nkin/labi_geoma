import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from copy import deepcopy as dc
import time
from sidrpy.classes import Point, Vector
from sidrpy.polygon_utils import point_location_from_convex_polygon, point_location_from_simple_polygon_octane
from sidrpy.utils import point_location_from_vec
from sidrpy.delete_me import calc_frames_number

start = time.time()

fig, ax = plt.subplots(1, 1)
fps = 30

x_outer = [0.3, 0.1, 0.15, 0.3, 0.5, 0.7, 0.85, 0.9, 0.75]
y_outer = [0.85, 0.6, 0.5, 0.3, 0.1, 0.3, 0.5, 0.6, 0.85]
points_outer = [Point(*coord) for coord in zip(x_outer, y_outer)]
lines_outer = [[points_outer[-1], points_outer[0]],
               *[[points_outer[i], points_outer[i + 1]] for i in range(len(points_outer) - 1)]]
x_inner = [0.35, 0.55, 0.6, 0.65, 0.75, 0.65, 0.6, 0.55]
y_inner = [0.6, 0.55, 0.35, 0.55, 0.6, 0.65, 0.75, 0.65]
points_inner = [Point(coord) for coord in zip(x_inner, y_inner)]

number_of_points = 100
points_anim = list()

while len(points_anim) != number_of_points:
    p0 = Point(*np.random.rand(2))  # Point(0.4, 0.4)
    if point_location_from_convex_polygon(x_outer, y_outer, p0) == "inside" and \
            point_location_from_simple_polygon_octane(x_inner, y_inner, p0) == "outside":
        points_anim.append(p0)

speed_const = 0.005
speed_vectors = list()
for _ in range(number_of_points):
    random_angle = np.random.rand() * 2 * np.pi  # np.pi
    speed_vectors.append(Vector(speed_const * np.cos(random_angle),
                                speed_const * np.sin(random_angle)))

points_and_vectors: list[dict[...]] = [dict(zip(["point", "vector"], el)) for el in zip(points_anim, speed_vectors)]
global_orientation = point_location_from_vec(points_anim[0], *lines_outer[0])

frames_num = calc_frames_number(dc(x_outer), dc(y_outer), dc(x_inner), dc(y_inner), dc(points_and_vectors))
def animate(i):
    # flag_list = [el["vector"] == Vector() for el in points_and_vectors]
    # if False not in flag_list:
    #     print("all point stuck!!!!", flag_list)
        # print(i)
    print(f"{i}/{frames_num}")
    ax.clear()
    ax.plot([*x_outer, x_outer[0]], [*y_outer, y_outer[0]], color='tab:green', marker='.')
    ax.plot([*x_inner, x_inner[0]], [*y_inner, y_inner[0]], color='tab:blue', marker='.')
    ax.scatter([point.x for point in points_anim], [point.y for point in points_anim], color='tab:orange', marker='.')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    for point_vec in points_and_vectors:
        point_vec["point"].x_arr += point_vec["vector"].x_arr
        point_vec["point"].y += point_vec["vector"].y
        if point_location_from_simple_polygon_octane(x_inner, y_inner, point_vec["point"]) == "inside" and\
                not point_vec["vector"] == Vector():
            point_vec["vector"] = Vector()
            # print("base")
        if point_location_from_convex_polygon(x_outer, y_outer, point_vec["point"]) in ["outside", "on polygon"]:
            for line in lines_outer:
                if point_location_from_vec(point_vec["point"], *line) != global_orientation:
                    # print(point_location_from_vec(point_vec["point"], *line))
                    p1, p2 = line
                    # ax.scatter([p1.x, p2.x], [p1.y, p2.y], color='tab:purple')
                    ax.plot([p1.x_arr, p2.x_arr], [p1.y, p2.y], color='tab:orange', marker=".")
                    v_i = point_vec["vector"]
                    q_j = Vector().from_points(*line)
                    point_vec["vector"] = 2 * ((v_i * q_j) / (q_j * q_j)) * q_j - v_i
                    break

# print(points_and_vectors)

print(f"{frames_num=}")
ani = FuncAnimation(fig, animate, frames=frames_num + fps // 10,
                    interval=50, repeat=False)
ani.save(f"points_and_polygons_{fps}fps.gif", dpi=300, writer=PillowWriter(fps=fps))
print(f"work time:\t{time.time() - start}")