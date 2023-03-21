import numpy as np
def are_on_the_same_line(p1: dict, p2: dict, p3: dict):
    if max(p1.get("x"), p3.get("x")) >= p2.get("x") >= min(p1.get("x"), p3.get("x")) and \
            max(p1.get("y"), p3.get("y")) >= p2.get("y") >= min(p1.get("y"), p3.get("y")):
        return True
    return False


def are_two_lines_intersect(p0: dict, p1: dict, p2: dict, p3: dict):
    det1 = np.linalg.det([[p3.get("x") - p2.get("x"), p3.get("y") - p2.get("y")],
                          [p0.get("x") - p2.get("x"), p0.get("y") - p2.get("y")]])
    det2 = np.linalg.det([[p3.get("x") - p2.get("x"), p3.get("y") - p2.get("y")],
                          [p1.get("x") - p2.get("x"), p1.get("y") - p2.get("y")]])
    det3 = np.linalg.det([[p1.get("x") - p0.get("x"), p1.get("y") - p0.get("y")],
                          [p2.get("x") - p0.get("x"), p2.get("y") - p0.get("y")]])
    det4 = np.linalg.det([[p1.get("x") - p0.get("x"), p1.get("y") - p0.get("y")],
                          [p3.get("x") - p0.get("x"), p3.get("y") - p0.get("y")]])
    if det1 == det2 == det3 == det3 == 0:
        if (are_on_the_same_line(p0, p2, p1) or are_on_the_same_line(p0, p3, p1)
                or are_on_the_same_line(p2, p0, p3) or are_on_the_same_line(p2, p1, p3)):
            return True
        else:
            return False
    return True if (det1 * det2 <= 0) and (det3 * det4 <= 0) else False


def is_polygon_simple(x: list, y: list):
    n = len(x)
    # x=[...], y=[...] -> coords = [{'x': x1, 'y': y1}, ..., {'x': xn, 'y': yn}, {'x': x1, 'y': y1}]
    coords = [dict(zip(["x", "y"], coord)) for coord in zip([*x, x[0]], [*y, y[0]])]
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