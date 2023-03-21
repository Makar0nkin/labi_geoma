from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from lab_2.utils import is_polygon_simple

@dataclass
class Point:
    x: int
    y: int

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


# координаты многоугольника и точки

x0, y0 = np.random.random(2) * 12
p0 = Point(x0, y0)

n = np.random.randint(4, 5 + 1)
x, y = np.random.rand(2, n) * 12
while not is_polygon_simple(x, y):
    n = np.random.randint(4, 5 + 1)
    x, y = np.random.rand(2, n) * 12

x_max: int = max(x)
x_min: int = min(x)

y_max: int = max(y)
y_min: int = min(y)

points = [Point(x[i], y[i]) for i in range(n)]


# построение многоугольника

plt.plot([*x, x[0]], [*y, y[0]])
plt.scatter(x0, y0)

# габаритный тест

if (x0 < x_min) or (x0 > x_max) or (y0 < y_min) or (y0 > y_max):
    plt.title('Снаружи')
else:
    # лучевой тест
    q = Point(x_min, y0)
    s: int = 0

    for i in range(n - 1):
        if are_intersected(points[i], points[i + 1], q, p0):
            if not is_point_on_line(points[i], q, p0) \
                    and not is_point_on_line(points[i + 1], q, p0):
                s += 1
            elif is_point_on_line(points[i], q, p0):
                k = 0
                while is_point_on_line(points[i + k], q, p0):
                    k += 1
                    if (not is_point_on_line(points[i - 1], q, p0) == is_point_on_line(points[i + k], q, p0)
                            and not is_point_on_line(points[i - 1], q, p0)
                            and not is_point_on_line(points[i + k], q, p0)):
                        s += 1

    plt.title("Снаружи" if s % 2 == 0 else "Внутри")

plt.show()
