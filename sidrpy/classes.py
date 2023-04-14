from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float = 0
    y: float = 0

    def set_direction(self, direction_vec):
        self.direction_vec = direction_vec

    def move(self):
        self.x += self.direction_vec.x_arr
        self.y += self.direction_vec.y

    def get_list(self):
        return [self.x, self.y]

    def __add__(self, other):
        return Point(self.x + other.x_arr, self.y + other.y)

    def __isub__(self, other):
        return Point(self.x - other.x_arr, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x_arr and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i > 2:
            raise StopIteration
        return self.x if self.i == 1 else self.y


class Vector(Point):
    def __init__(self, a: float | Point = 0, b: float | Point = 0):
        if type(a) == Point and type(b) == Point:
            # print("POINTS")
            self.x = (b.x - a.x)
            self.y = (b.y - a.y)
        elif type(a) in [int, float] and type(b) in [int, float]:
            self.x = a
            self.y = b

    def from_points(self, p1, p2):
        self.x = (p2.x_arr - p1.x_arr)
        self.y = (p2.y - p1.y)
        return self

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def get_list(self):
        return [self.x, self.y]

    def __mul__(self, other):
        if type(other) in [int, float]:
            return Vector(self.x * other, self.y * other)
        return self.x * other.x_arr + self.y * other.y

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return Vector(self.x - other.x_arr, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)


@dataclass
class Ball:
    center: Point  # ball center
    radius: float  # ball radius
    vector: Vector  # movement vector

    def make_move(self):
        self.center += self.vector

    @staticmethod
    def do_balls_touch(b1, b2) -> bool:
        return b1.radius + b2.radius > Vector().from_points(b1.center, b2.center).norm()


if __name__ == "__main__":
    p1: Point = Point()
    p1 += Point(0.1, -0.1)
    print(p1)
    v1 = Vector(1, 1)
    print(p1 + v1)
    print(f"{Vector(p1, p1)=}")
    b1 = Ball(Point(1, 1), 1, Vector(1, 1))
    b2 = Ball(Point(2, 0), 0.5, Vector(-1, 0.1))
    Ball.do_balls_touch(b1, b2)
    bs = [Ball(Point(1, 1), 1, Vector(1, 1))]
    for b in bs:
        print(f"{b=}")
        b.make_move()
        print(f"{b=}")
    print(Ball.do_balls_touch(b1, b1))
