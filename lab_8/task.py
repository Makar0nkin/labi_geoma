from sidrpy.classes import Vector, Point
from sidrpy.utils import are_two_lines_intersect, point_location_from_vec
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from ing_theme_matplotlib import mpl_style

mpl_style(True)

def get_intersection_point(p1, p2, p3, p4):
    n = Vector(-(p2.y - p1.y), p2.x - p1.x)
    t = (n * Vector(p3, p1)) / (n * Vector(p3, p4))
    p = Vector(p3, p4)
    return Point(p3.x + t * p.x, p3.y + t * p.y)

def det_four_points(p1, p2, p3, p4):
    return np.linalg.det([[p2.x - p1.x, p2.y - p1.y], [p4.x - p3.x, p4.y - p3.y]])

def is_aimed(a1, a2, b1, b2):
    are_collinear = point_location_from_vec(b1, b2, a1) == "on line" and point_location_from_vec(b1, b2, a2) == "on line"

    if are_collinear:
        # если a1a2 и b1b2 не пересекаются и a2, b1 по одну сторону от a1 - нацелен
        if (not are_two_lines_intersect(a1, a2, b1, b2)) and ((a2.x - a1.x) * (b1.x - a1.x) + (a2.y - a1.y) * (b1.y - a1.y) > 0):
            return True
    else:
        if det_four_points(b1, b2, a1, a2) < 0 and det_four_points(b1, b2, b1, a2) > 0:
            return True
        elif det_four_points(b1, b2, a1, a2) > 0 and det_four_points(b1, b2, b1, a2) < 0:
            return True
    return False

def polygon_intersection(P, Q):
    n = len(P)
    m = len(Q)
    Res = []  # массив точек пересечения

    p = 0  # "окно" для движения по первому многоугольнику
    q = 0  # "окно" для движения по второму многоугольнику

    # фиксируем окно q из второго многоугольника и для него подбираем окно p из первого многоугольника
    for i in range(0, n):
        # справа
        if point_location_from_vec(P[i], P[(i + 1) % n], Q[0]) == "right" or point_location_from_vec(Q[0], Q[1], P[(i + 1) % n]) == "right":
            p = i
            break

    # следующие точки
    p_next = (p + 1) % n
    q_next = (q + 1) % m

    # цикл по точкам многоугольника
    for i in range(0, 2 * (n + m)):

        # 1. если окна нацелены друг на друга
        if is_aimed(P[p], P[p_next], Q[q], Q[q_next]) and is_aimed(Q[q], Q[q_next], P[p], P[p_next]):
            # двигаем внешнее окно
            if point_location_from_vec(Q[q], Q[q_next], P[p_next]) == "right": # is_external(P[p], P[p_next], Q[q], Q[q_next]):
                p = p_next
                p_next = (p + 1) % n
            else:
                q = q_next
                q_next = (q + 1) % m

        # 2. если p нацелен на q, а q на p не нацелен
        elif is_aimed(P[p], P[p_next], Q[q], Q[q_next]) and not is_aimed(Q[q], Q[q_next], P[p], P[p_next]):
            # если p - внешнее окно, то добавляем конечную вершину в ответ
            if not point_location_from_vec(Q[q], Q[q_next], P[p_next]) == "right":  # not is_external(P[p], P[p_next], Q[q], Q[q_next]):
                Res.append(P[p_next])
            # двигаем окно p
            p = p_next
            p_next = (p + 1) % n

        # 3. если q нацелен на p, а p на q не нацелен
        elif not is_aimed(P[p], P[p_next], Q[q], Q[q_next]) and is_aimed(Q[q], Q[q_next], P[p], P[p_next]):
            # если q - внешнее окно, то добавляем конечную вершину в ответ
            if not point_location_from_vec(P[p], P[p_next], Q[q_next]) == "right":  # is_external(Q[q], Q[q_next], P[p], P[p_next]):
                Res.append(Q[q_next])
            # двигаем окно q
            q = q_next
            q_next = (q + 1) % m

        # 4. если окна не нацелены друг на друга
        elif not is_aimed(P[p], P[p_next], Q[q], Q[q_next]) and not is_aimed(Q[q], Q[q_next], P[p], P[p_next]):
            # если окна пересекаются, то добавляем точку пересечения в ответ
            if are_two_lines_intersect(P[p], P[p_next], Q[q], Q[q_next]):
                Res.append(get_intersection_point(P[p], P[p_next], Q[q], Q[q_next]))
            # двигаем внешнее окно
            if point_location_from_vec(Q[q], Q[q_next], P[p_next]) == "right":  # is_external(P[p], P[p_next], Q[q], Q[q_next]):
                p = p_next
                p_next = (p + 1) % n
            else:
                q = q_next
                q_next = (q + 1) % m

        # если первая добавленная точка совпала с последней - выход
        if len(Res) > 1 and Res[0] == Res[-1]:
            del Res[-1]
            break

    return Res

def get_intersection_parameter(segment, p1, p2):
    n = Vector(-(p2.y - p1.y), p2.x - p1.x)
    return (n * Vector(segment[0], p1)) / (n * Vector(segment[0], segment[1]))

def get_intersection_type(p1, p2, segment):
    normal_vector = Vector(p2.y - p1.y, -(p2.x - p1.x))
    segment_vector = Vector(segment[1].x - segment[0].x, segment[1].y - segment[0].y)

    if segment_vector * normal_vector > 0:
        return 1  # п покид
    elif segment_vector * normal_vector < 0:
        return -1  # п вход
    else:
        return 0


def cyrus_beck(segment, polygon_points):

    t0_values = [0]
    t1_values = [1]
    n = len(polygon_points)

    for i in range(n):
        intersection_type = get_intersection_type(polygon_points[i], polygon_points[(i + 1) % n], segment)
        t = get_intersection_parameter(segment, polygon_points[i],
                                       polygon_points[(i + 1) % n])  # параметр точки пересечения
        if intersection_type == -1:
            t0_values.append(t)
        elif intersection_type == 1:
            t1_values.append(t)
        else:
            continue

    t0 = max(t0_values)
    t1 = min(t1_values)

    if t0 <= t1:
        x1 = segment[0].x + (segment[1].x - segment[0].x) * t0  # xA + (xB-xA)*t0
        x2 = segment[0].x + (segment[1].x - segment[0].x) * t1
        y1 = segment[0].y + (segment[1].y - segment[0].y) * t0
        y2 = segment[0].y + (segment[1].y - segment[0].y) * t1
        return Point(x1, y1), Point(x2, y2)
    else:
        return segment[0], segment[0]  # если пустое множество, то возвращаем две совпадающие точки


def main_task(P, Q):
    fig, ax = plt.subplots()
    camera = Camera(fig)

    count = 50
    for k in range(0, count):
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        # рисуем многоугольники
        ax.plot([p.x for p in [*P, P[0]]], [p.y for p in [*P, P[0]]], color="tab:grey")
        ax.plot([p.x for p in [*Q, Q[0]]], [p.y for p in [*Q, Q[0]]], color="tab:grey")
        ax.fill([p.x for p in [*P, P[0]]], [p.y for p in [*P, P[0]]], color="tab:blue")
        ax.fill([p.x for p in [*Q, Q[0]]], [p.y for p in [*Q, Q[0]]], color="tab:orange")

        res = polygon_intersection(P, Q)  # массив точек пересечения многоугольников
        ax.fill(list(map(lambda p: p.x, res)), list(map(lambda p: p.y, res)), "tab:purple")

        p0_new, p2_new = cyrus_beck([P[0], P[2]], Q)  # концевые точки отсечения отрезка Р0Р2
        ax.plot([p.x for p in [p0_new, p2_new]], [p.y for p in [p0_new, p2_new]], color="tab:grey")

        # делаем шаг анимации
        camera.snap()

        # двигаем многоугольники
        for p1 in P:
            p1.move()
        for p2 in Q:
            p2.move()

        plt.draw()
        plt.gcf().canvas.flush_events()
        # sleep(0.00001)

    # сохраняем анимацию
    animation = camera.animate()
    animation.save('polygon_intersection_huema_+1.gif')

P = [Point(0.1, 0.1), Point(0.3, 0.1), Point(0.4, 0.4), Point(0.2, 0.4)]
Q = [Point(0.7, 0.25), Point(0.9, 0.3), Point(0.85, 0.55), Point(0.6, 0.59), Point(0.55, 0.3)]

# задаем направление и скорость точкам
speed = 0.01
for point in P:
  point.set_direction(Vector(speed * 1, 0), speed)
for point in Q:
  point.set_direction(Vector(speed * -1, 0), speed)

main_task(P, Q)