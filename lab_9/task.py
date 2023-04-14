import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from matplotlib.animation import FuncAnimation, FFMpegWriter
from copy import deepcopy
from sidrpy.classes import Point, Vector, Ball
from ing_theme_matplotlib import mpl_style

mpl_style(True)


def split_to_sorted(ps: list[Point]) -> tuple[list[Point], list[Point]]:
    return sorted(ps, key=lambda p: p.x_arr), sorted(ps, key=lambda p: p.y)


def nearest_pair(points_x: list[Point], points_y: list[Point]) -> dict[str, tuple[Point, Point] | float]:
    if len(points_x) == 2:
        return {"points": tuple(points_x), "distance": Vector().from_points(*points_x).norm()}
    elif len(points_x) == 3:
        return {"points": (
            ps := tuple(sorted(combinations(points_x, 2), key=lambda cpl: Vector().from_points(*cpl).norm())[0])),
            "distance": Vector().from_points(*ps).norm()
        }
    else:
        sep = int(len(points_x) / 2)
        points_x_left: list[Point] = points_x[:sep]
        points_x_right: list[Point] = points_x[sep:]
        points_y_left: list[Point] = []
        points_y_right: list[Point] = []
        point_flag = True
        for p in points_y:
            if p.x < points_x[sep].x:
                points_y_left.append(p)
            elif p.x > points_x[sep].x:
                points_y_right.append(p)
            else:
                points_y_left.append(p) if point_flag else points_y_right.append(p)
                point_flag = not point_flag

        delta_left: dict = nearest_pair(points_x_left, points_y_left)
        delta_right: dict = nearest_pair(points_x_right, points_y_right)
        res = min(delta_left, delta_right, key=lambda dc: dc.get("distance"))
        # print(f"{len(delta_left.get('points'))=}\n{len(delta_right.get('points'))=}\n{len(res.get('points'))=}")

        points_y_delta: list[Point] = []
        for p in points_y:
            if abs(p.x - points_x[sep].x) <= res["distance"]:
                points_y_delta.append(p)
        observed_points_num = 7
        # iterable_n = len(points_y_delta) if len(points_y_delta) <= observed_points_num else observed_points_num

        for i in range(len(points_y_delta) - 1):
            for j in range(i + 1, min((i + observed_points_num + 1), len(points_y_delta))):
                if (new_delta := Vector().from_points(points_y_delta[i], points_y_delta[j]).norm()) <= res["distance"]:
                    res["distance"] = new_delta
                    res["points"] = (points_y_delta[i], points_y_delta[j])
        return res


fig, ax = plt.subplots()
fps = 30

n = 10
# points = [Point(*coord) for coord in zip(*np.random.rand(2, n))]
radius = 0.04
# radii = [radius for _ in range(n)]  # np.random.randint(1, 5) / 100
speed_const = 0.005
vecs = list()
for _ in range(n):
    random_angle = np.random.rand() * 2 * np.pi  # np.pi
    vecs.append(Vector(speed_const * np.cos(random_angle),
                       speed_const * np.sin(random_angle)))
# balls = [Ball(*data) for data in zip(points, [radius for _ in range(n)], vecs)]

balls = [Ball(center=Point(*np.random.rand(2)), radius=radius, vector=vecs[0])]
i: int = 1
while i < n:
    new_ball = Ball(center=Point(*np.random.rand(2)), radius=radius, vector=vecs[i])
    for b1, b2 in combinations([*balls, new_ball], 2):
        # print(b1, b2)
        if Ball.do_balls_touch(b1, b2):
            break
    else:
        i += 1
        balls.append(new_ball)
        # print(f"{balls=}")

# print(balls)
x1, y1, x2, y2 = -0.05, -0.05, 1.05, 1.05  # np.random.rand(4)
rect_ps = [Point(x1, y1), Point(x1, y2), Point(x2, y2), Point(x2, y1)]
recs_ps_last_dubl = deepcopy(rect_ps)
recs_ps_last_dubl.append(rect_ps[0])
rect_vecs = [Vector().from_points(recs_ps_last_dubl[i], recs_ps_last_dubl[i + 1]) for i in
             range(len(recs_ps_last_dubl) - 1)]
print(rect_vecs)

frames_n = 100
def animate(i: int):
    print(f"{i}/{frames_n}")
    ax.clear()
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.1)
    nearest_points = nearest_pair(*split_to_sorted([b.center for b in balls]))
    if nearest_points.get("distance") < 2 * radius:
        for i in [i for i, b in enumerate(balls) if b.center in nearest_points.get("points")]:
            balls[i].vector = -balls[i].vector

    for b in balls:
        q_j = None
        if b.center.x - radius < x1:
            q_j = rect_vecs[0]
        elif b.center.y + radius > y2:
            q_j = rect_vecs[1]
        elif b.center.x + radius > x2:
            q_j = rect_vecs[2]
        elif b.center.y - radius < y1:
            q_j = rect_vecs[3]

        if q_j is not None:
            # v_i = b.vector
            b.vector = 2 * ((b.vector * q_j) / (q_j * q_j)) * q_j - b.vector
        b.make_move()
        circle = plt.Circle((b.center.x, b.center.y), b.radius, color="tab:blue")
        ax.add_patch(circle)
    ax.plot([p.x for p in recs_ps_last_dubl], [p.y for p in recs_ps_last_dubl])


anime = FuncAnimation(fig, animate, frames=frames_n, interval=50, repeat=False)
anime.save(f"big_balls_{fps}fps.mp4", dpi=300, writer=FFMpegWriter(fps=fps))