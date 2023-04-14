import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt
from sidrpy.utils import point_location_from_vec
from sidrpy.classes import Point, Vector

fig = plt.figure()
camera = Camera(fig)


def find_middle_point(p1: Point, p2: Point, p3: Point):
    if Vector().from_points(p3, p1) * Vector().from_points(p3, p2) <= 0:
        return p3
    elif Vector().from_points(p2, p1) * Vector().from_points(p2, p3) <= 0:
        return p2
    else:
        return p1


def dynamic_convex_hull(new_points: list, prev_convex_hull: list):
    if len(new_points) <= 2:
        return new_points
    if len(new_points) == 3:
        if point_location_from_vec(new_points[0], new_points[1], new_points[2]) == "on line":
            mid_point = find_middle_point(new_points[0], new_points[1], new_points[2])
            new_points.remove(mid_point)
            return new_points
        else:
            if point_location_from_vec(new_points[2], new_points[0], new_points[1]) == "left":
                return [new_points[0], new_points[1], new_points[2]]
            if point_location_from_vec(new_points[2], new_points[0], new_points[1]) == "right":
                return [new_points[0], new_points[2], new_points[1]]

    if len(new_points) > 3:
        new_point = new_points[-1]
        start = -1
        end = -1
        if point_location_from_vec(new_point, prev_convex_hull[-1], prev_convex_hull[0]) == "right" \
                and point_location_from_vec(new_point, prev_convex_hull[0], prev_convex_hull[1]) == "right":
            for i in range(len(prev_convex_hull) - 1):
                if point_location_from_vec(new_point, prev_convex_hull[i], prev_convex_hull[i + 1]) == "right":
                    start = i + 1
            end = len(prev_convex_hull) - 1
            for i in range(len(prev_convex_hull) - 1, start, -1):
                if point_location_from_vec(new_point, prev_convex_hull[i - 1], prev_convex_hull[i]) == "right":
                    end = i - 1
            return [*prev_convex_hull[start:end + 1], new_point]
        else:
            prev_convex_hull.append(prev_convex_hull[0])
            for i in range(len(prev_convex_hull) - 1):
                if point_location_from_vec(new_point, prev_convex_hull[i], prev_convex_hull[i + 1]) == "right":
                    start = i
                    break
            if start == -1:
                prev_convex_hull.pop()
                return prev_convex_hull
            for i in range(start, len(prev_convex_hull) - 1):
                if point_location_from_vec(new_point, prev_convex_hull[i], prev_convex_hull[i + 1]) == "right":
                    end = i + 1
                else:
                    break
            prev_convex_hull.pop()
            return [*prev_convex_hull[0:start + 1], new_point, *prev_convex_hull[end:len(prev_convex_hull)]]


extend = []  # [Point(0, 0.1), Point(0, 0.2), Point(0, 0.3), Point(0, 0.4), Point(0, 0.5)]
points = []
convex_hull = []

n = 100
for i in range(n):
    print(f"{i + 1}/{n}")
    if extend:
        points.append(extend.pop())
    else:
        points.append(Point(*np.random.rand(2)))
    convex_hull = dynamic_convex_hull(points, convex_hull)
    plt.scatter([p.x for p in points], [p.y for p in points], color="tab:orange")
    plt.scatter([p.x_arr for p in convex_hull], [p.y for p in convex_hull], color="tab:blue")
    plt.plot([p.x_arr for p in [*convex_hull, convex_hull[0]]], [p.y for p in [*convex_hull, convex_hull[0]]], color="tab:blue")
    camera.snap()

animation = camera.animate(blit=False, interval=200)
animation.save("animation.mp4")
