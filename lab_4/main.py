import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from copy import deepcopy as dc
import time
import math
import random
import functools
from celluloid import Camera
from utils import point_location_from_vec
from bad_classes import Point, Vector

fig = plt.figure()
camera = Camera(fig)


def init_points():
    points = [Point(random.random() * 10, random.random() * 10) for _ in range(0, 10)]
    return points


def compare(p1, p2):
    if (p1.y < p2.y):
        return -1
    elif (p1.y > p2.y):
        return 1
    else:
        return 0


def sortPoints(points):
    return sorted(points, key=functools.cmp_to_key(compare))


# косинус угла между векторами
def cos(v1: Vector, v2: Vector):
    return (v1 * v2) / (v1.get_length() * v2.get_length())


# находим полярные углы
def get_arcs(points: list):
    arcs = []
    s0 = points[0]
    # полярная ось
    i_vector = Vector(Point(0, 0), Point(1, 0))

    for i in range(1, len(points)):
        v = Vector(s0, points[i])
        arc_value = math.acos(cos(v, i_vector))
        arcs.append(arc_value)

    for i in range(len(arcs)):
        min_idx = i
        for j in range(i + 1, len(arcs)):
            if arcs[min_idx] > arcs[j]:
                min_idx = j
                # если одинаковый полярный угол, оставляем ту, которая дальше
            elif arcs[min_idx] == arcs[j]:
                min_idx_vector = Vector(s0, points[min_idx])
                j_vector = Vector(s0, points[j])
                if j_vector.get_length() > min_idx_vector.get_length():
                    min_idx = j

        arcs[i], arcs[min_idx] = arcs[min_idx], arcs[i]
        points[i + 1], points[min_idx + 1] = points[min_idx + 1], points[i + 1]
    return arcs


# построение выпуклой оболочки
def build_convex_hull(points: list):
    arcs = get_arcs(points)

    convex_hull_points = []

    for i in range(len(points)):
        while (len(convex_hull_points) >= 2) and (
                point_location_from_vec(points[i], convex_hull_points[-2], convex_hull_points[-1]) == "right"):
            del convex_hull_points[-1]
            draw_points(points)
            draw_convex_hull(convex_hull_points, "green")
        convex_hull_points.append(points[i])
        draw_points(points)
        draw_convex_hull(convex_hull_points, "green")

    draw_points(points)
    convex_hull_points.append(convex_hull_points[0])
    draw_convex_hull(convex_hull_points, "green")
    convex_hull_points.pop()


def draw_convex_hull(convex_hull_points: list, color: str):
    for i in range(len(convex_hull_points) - 1):
        plt.plot([convex_hull_points[i].x_arr, convex_hull_points[i + 1].x_arr],
                 [convex_hull_points[i].y, convex_hull_points[i + 1].y], color=color)
    camera.snap()


def draw_points(points: list):
    plt.scatter([p.x_arr for p in points], [p.y for p in points], color="tab:blue")
    # for i in range(len(points)):
    #     plt.scatter(points[i].x, points[i].y)


points = sortPoints(init_points())
draw_points(points)
build_convex_hull(points)

plt.grid(True)
animation = camera.animate(blit=False, interval=300)
animation.save("animation.gif")
plt.show()
