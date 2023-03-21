from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Point:
    x: int
    y: int


# координаты многоугольника и точки

x0, y0 = np.random.random(2) * 12
p0 = Point(x0, y0)

# x = [2, 10, 3, 2, 2]  # координаты вершин многоугольника
# y = [3, 2, 5, 10, 3]
x = [4, 6, 8, 4]
y = [3, 9, 4, 3]
n = 4

x_max: int = max(x)
x_min: int = min(x)

y_max: int = max(y)
y_min: int = min(y)

q = Point(x_min, y0)

points = [Point(x[i], y[i]) for i in range(n)]


def are_intersected(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    d1 = np.linalg.det([[p4.x - p3.x, p4.y - p3.y], [p1.x - p3.x, p1.y - p3.y]])
    d2 = np.linalg.det([[p4.x - p3.x, p4.y - p3.y], [p2.x - p3.x, p2.y - p3.y]])
    d3 = np.linalg.det([[p2.x - p1.x, p2.y - p1.y], [p3.x - p1.x, p3.y - p1.y]])
    d4 = np.linalg.det([[p2.x - p1.x, p2.y - p1.y], [p4.x - p1.x, p4.y - p1.y]])

    if d1 * d2 <= 0 and d3 * d4 <= 0:
        return True
    else:
        return False


def is_point_on_line(p1: Point, p2: Point, p3: Point) -> bool:
    d = np.linalg.det([[p3.x - p2.x, p3.y - p2.y], [p1.x - p2.x, p1.y - p2.y]])
    if d > 0:
        return False
    elif d < 0:
        return False
    else:
        return True


# построение многоугольника

plt.plot(x, y)
plt.scatter(x0, y0)

# габаритный тест

if (x0 < x_min) or (x0 > x_max) or (y0 < y_min) or (y0 > y_max):
    plt.title('Снаружи')
else:
    # лучевой тест

    s: int = 0
    for i in range(n - 1):
        if are_intersected(points[i], points[i + 1], q, p0):
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

                if not is_point_on_line(points[j], q, p0) == is_point_on_line(points[k], q, p0
                                                                                            and not is_point_on_line(
                    points[j], q, p0)
                                                                                            and not is_point_on_line(
                    points[k], q, p0)):
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
                if (not is_point_on_line(points[j], q, p0) == is_point_on_line(points[k], q, p0)
                        and not is_point_on_line(points[j], q, p0)
                        and not is_point_on_line(points[k], q, p0)):
                    s += 1
                i = k

        plt.title("Снаружи" if s % 2 == 0 else "Внутри")

plt.show()
