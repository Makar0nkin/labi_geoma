import numpy as np
from sidrpy.classes import Point, Vector
from sidrpy.utils import are_two_lines_intersect, calc_angle, point_location_from_vec, is_point_on_line, calc_octane, calc_area

def perimeter(points: list):
    hull_perimeter = 0
    for i in range(len(points) - 1):
        hull_perimeter += Vector(points[i], points[i + 1]).get_length()
    hull_perimeter += Vector(points[len(points) - 1], points[0]).get_length()
    return hull_perimeter

def is_polygon_simple(x: list, y: list):
    n = len(x)
    # x=[...], y=[...] -> coords = [{'x': x1, 'y': y1}, ..., {'x': xn, 'y': yn}, {'x': x1, 'y': y1}]
    # coords = [dict(zip(["x", "y"], coord)) for coord in zip([*x, x[0]], [*y, y[0]])]
    coords = [Point(*coord) for coord in zip([*x, x[0]], [*y, y[0]])]
    lines = [[coords[i], coords[i + 1]] for i in range(n)]
    is_polygon_simple_flag = True
    for i in range(n):
        strange_lines: list[...] = lines.copy()
        current_line: dict
        # берем текущую линию, выносим в переменную и удаляем ее соседей
        if i == 0:
            strange_lines.pop(i + 1)
            strange_lines.pop(-1)
            current_line = strange_lines.pop(0)
        elif i == n - 1:
            strange_lines.pop(i - 1)
            strange_lines.pop(0)
            current_line = strange_lines.pop(-1)
        else:
            strange_lines.pop(i + 1)
            strange_lines.pop(i - 1)
            current_line = strange_lines.pop(i - 1)

        line_combinations = [[current_line, strange_line] for strange_line in strange_lines]
        # Если какие-то две не соседние линии пересекаются, то многоугольник не простой
        for line_combination in line_combinations:
            points = [point for line in line_combination for point in line]
            if are_two_lines_intersect(*points):
                is_polygon_simple_flag = False
                break
        if not is_polygon_simple_flag:
            break

    return True if is_polygon_simple_flag else False

def point_location_from_simple_polygon_octane(x, y, p0):
    points = [Point(*coord) for coord in zip([*x, x[0]], [*y, y[0]])]
    n = len(points) - 1
    s = 0
    for i in range(n):
        sigma_i = calc_octane(Vector().from_points(p0, points[i]))
        sigma_i_1 = calc_octane(Vector().from_points(p0, points[i + 1]))
        if sigma_i_1 == None:
            return "on polygon"
        delta_i = sigma_i_1 - sigma_i
        if delta_i > 4:
            delta_i -= 8
        elif delta_i < -4:
            delta_i += 8
        elif np.abs(delta_i) == 4:
            d = np.linalg.det([[points[i].x - p0.x_arr, points[i].y - p0.y],
                               [points[i + 1].x - p0.x_arr, points[i + 1].y - p0.y]])
            if d > 0:
                delta_i = 4
            elif d < 0:
                delta_i = -4
            elif d == 0:
                return "on polygon"

        s += delta_i

    if np.abs(s) == 8:
        return "inside"
    elif s == 0:
        return "outside"



def point_location_from_convex_polygon(x, y, p0):
    points = [Point(*coord) for coord in zip([*x, x[0]], [*y, y[0]])]
    z = Point(np.mean(x), np.mean(y))  # Point(2.5, 1)
    n = len(points) - 1

    start = 0
    end = n
    while end - start > 1:
        sep = int((start + end) / 2)
        if calc_angle(points[0], z, p0) < calc_angle(points[0], z, points[sep]):
            end = sep
        else:
            start = sep

    # plt.scatter([points[start].x, points[end].x], [points[start].y, points[end].y])
    if point_location_from_vec(p0, points[start], points[end]) == point_location_from_vec(z, points[start], points[end]):
        return "inside"
    elif point_location_from_vec(p0, points[start], points[end]) == "on line":
        return "on polygon"
    else:
        return "outside"

def point_location_from_simple_polygon(x, y, p0):
    n = 4

    x_max: int = max(x)
    x_min: int = min(x)

    y_max: int = max(y)
    y_min: int = min(y)
    x0, y0 = p0.x_arr, p0.y

    q = Point(x_min - 1, y0)

    points = [Point(*coord) for coord in zip([*x, x[0]], [*y, y[0]])]
    # габаритный тест
    if (x0 < x_min) or (x0 > x_max) or (y0 < y_min) or (y0 > y_max):
        return "outside"
        # plt.title('Снаружи')
    else:
        # лучевой тест
        s: int = 0
        for i in range(n - 1):
            if are_two_lines_intersect(points[i], points[i + 1], q, p0):
                if not is_point_on_line(points[i], q, p0) \
                        and not is_point_on_line(points[i + 1], q, p0):
                    s += 1
                elif is_point_on_line(points[i], q, p0):
                    j = i - 1
                    while is_point_on_line(points[j], q, p0):
                        j -= 1
                        if j < 0:
                            j += len(points) - 1
                    k = (i + 1) % len(points)
                    while is_point_on_line(points[k], q, p0):
                        k += 1
                        if k >= len(points):
                            k -= len(points)

                    if not is_point_on_line(points[j], q, p0) == is_point_on_line(points[k], q, p0):
                        s += 1
                    i = k

                elif is_point_on_line(points[i + 1], q, p0) and not is_point_on_line(points[i], q, p0):
                    j = i
                    while is_point_on_line(points[j], q, p0):
                        j -= 1
                        if j < 0:
                            j += len(points) - 1
                    k = (i + 2) % len(points)
                    while is_point_on_line(points[k], q, p0):
                        k += 1
                        if k >= len(points):
                            k -= len(points)
                    if not is_point_on_line(points[j], q, p0) == is_point_on_line(points[k], q, p0):
                        s += 1
                    i = k

            # plt.title("Снаружи" if s % 2 == 0 else "Внутри")
            return "outside" if s % 2 == 0 else "inside"


def convex_hull_jarvis(x: list, y: list) -> list[Point]:
    points = [Point(*coord) for coord in zip(x, y)]

    active_p = min(points, key=lambda p: p.y)
    epsilon = 0.1
    is_axis_direction_right = True
    axis_p: Point
    convex_hull = []
    while True:
        convex_hull.append(active_p)
        axis_p = Point(*active_p.get_list())
        axis_p.x += epsilon if is_axis_direction_right else -epsilon
        remain_points = list(
            filter(lambda p: (p.y > active_p.y if is_axis_direction_right else p.y < active_p.y), points))
        angles_rem_points = [{"point": p, "angle": calc_angle(axis_p, active_p, p)} for p in remain_points]
        if not angles_rem_points:
            is_axis_direction_right = False
            continue
        active_p = min(angles_rem_points, key=lambda el: el.get("angle")).get("point")
        if active_p == convex_hull[0]:
            break

    for i in range(len(convex_hull) - 1):
        if convex_hull[i] == convex_hull[i + 1]:
            convex_hull.pop(i)
            break

    return convex_hull

def calc_diameter(points):
    k = len(points)
    i = 1
    d = 0
    result = dict()
    while calc_area(points[-1], points[0], points[i - 1]) < \
            calc_area(points[-1], points[0], points[i]):
        i += 1
    start = i
    j = 0
    while start < k:
        tmp = start
        while calc_area(points[j % k], points[(j + 1) % k], points[tmp % k]) <= calc_area(points[j % k], points[(j + 1) % k], points[(tmp + 1) % k]):
            tmp += 1
        end = tmp
        for l in range(start, end + 1):
            if d < (d_new := Vector().from_points(points[j % k], points[l % k]).norm()):
                d = d_new
                result["points"] = [points[j % k], points[l % k]]
                result["diam"] = d
        start = end
        j += 1
    return result