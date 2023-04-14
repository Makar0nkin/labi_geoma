import numpy as np
import math
import random
from matplotlib import pyplot as plt
from celluloid import Camera
from sidrpy.classes import Point,Vector
from sidrpy.utils import point_location_from_vec
from sidrpy.polygon_utils import perimeter
from sidrpy.utils import vector_product


fig = plt.figure()
camera = Camera(fig)


def init_points():
    n = 15
    # x, y = np.random.rand(2, n)
    x = [random.randint(0, 10) for _ in range(15)]
    y = [random.randint(0, 10) for _ in range(15)]
    points = [Point(*coord) for coord in zip(x, y)]
    return points


# Разбиваем на множества Pl и Pr
def find_righter_points(points: list, p1: Point, p2: Point):
    righter_points = []
    for i in range(len(points)):
        if point_location_from_vec(points[i], p1, p2) == "right":
            righter_points.append(points[i])
    return righter_points



def find_lefter_points(points: list, p1: Point, p2: Point):
    lefter_points = []
    for i in range(len(points)):
        if point_location_from_vec(points[i], p1, p2) == "left":
            lefter_points.append(points[i])
    return lefter_points




def quick_hull(pL: Point, pR: Point, points: list, convex_hull_points: list):
    max_area_triangle = vector_product(pL, pR, points[0])
    s = points[0]
    for i in range(1, len(points)):
        # находим наиболее удаленную точку s
        if vector_product(pL, pR, points[i]) > max_area_triangle:
            max_area_triangle = vector_product(pL, pR, points[i])
            s = points[i]
    # теперь для каждой из сторон pls и spr находим точки левее
    S1 = find_lefter_points(points, pL, s)
    S2 = find_lefter_points(points, s, pR)
    # и если множество точек левее pls не пусто то повторяем quick hull и тд
    if S1:
        quick_hull(pL, s, S1, convex_hull_points)
    convex_hull_points.append(s)
    # и если множество точек левее spr не пусто то повторяем quick hull и тд
    if S2:
        quick_hull(s, pR, S2, convex_hull_points)


def complete_convex_hull(points: list):
    pl = max(points, key=lambda p: p.y)
    pr = min(points, key=lambda p: p.y)
    convex_hull_points = []

    lefter_points = find_lefter_points(points, pl, pr)
    righter_points = find_righter_points(points, pl, pr)

    convex_hull_points.append(pl)
    quick_hull(pl, pr, lefter_points, convex_hull_points)
    convex_hull_points.append(pr)
    quick_hull(pr, pl, righter_points, convex_hull_points)

    convex_hull_points.append(convex_hull_points[0])
    return convex_hull_points

# Анимация:

def draw_points(points: list):
    for i in range(len(points)):
        # draw_point(points[i])
        plt.scatter(points[i].x_arr, points[i].y, color='black')


def draw_convex_hull(convex_hull_points: list, color: str):
    for i in range(len(convex_hull_points) - 1):
        plt.plot([convex_hull_points[i].x_arr, convex_hull_points[i + 1].x_arr],
                 [convex_hull_points[i].y, convex_hull_points[i + 1].y], color=color)


def move(moving_points: list, vectors: list):
    for i in range(len(moving_points)):
        moving_points[i] = moving_points[i] + vectors[i]


def opposite_vectors_of_moving(vectors: list):
    for i in range(len(vectors)):
        vectors[i] = Point(-vectors[i].x_arr, -vectors[i].y)
    return vectors


def init_vectors_of_moving(points: list):
    vectors = []
    xs = [random.randint(-1, 1) for _ in range(len(points))]
    ys = [random.randint(-1, 1) for _ in range(len(points))]
    for i in range(len(xs)):
        p = Point(xs[i], ys[i])
        while p.x == 0 and p.y == 0:
            p = Point(random.randint(-1, 1), random.randint(-1, 1))
        vectors.append(p)
    return vectors

def init_motion(points: list):
    vectors = init_vectors_of_moving(points)
    PERIMETER_LIMIT = 100

    i = 0
    while i < 70:
        convex_hull_points = complete_convex_hull(points)

        draw_points(points)
        draw_convex_hull(convex_hull_points, "blue")
        camera.snap()

        if perimeter(convex_hull_points) >= PERIMETER_LIMIT:
            vectors = opposite_vectors_of_moving(vectors)

        # for i in range(len(points)):
        #     for j in range(len(convex_hull_points)):
        #         if points[i]==convex_hull_points[j] and perimeter(convex_hull_points) >= PERIMETER_LIMIT:
        #             points[i] = points[i] + opposite_vectors_of_moving(vectors)[i];
        #         else:
        #             points[i] = points[i] + vectors[i];
        move(points, vectors)
        i += 1



points = init_points()
init_motion(points)
plt.grid(True)
animation = camera.animate(blit=False, interval=300)
animation.save("animation.gif")
plt.show()



