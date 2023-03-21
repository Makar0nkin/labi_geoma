from dataclasses import dataclass

import numpy as np


@dataclass
class Point:
    x: float = 0
    y: float = 0

    def get_list(self):
        return [self.x, self.y]

    def __add__(self, other):
        # if type(other) == Vector:
        #     return Point
        return Point(self.x + other.x, self.y + other.y)

    def __isub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

@dataclass
class Vector(Point):
    def from_points(self, p1, p2):
        self.x = (p2.x - p1.x)
        self.y = (p2.y - p1.y)
        return self

    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def get_list(self):
        return [self.x, self.y]

    def __mul__(self, other):
        if type(other) in [int, float]:
            return Vector(self.x * other, self.y * other)
        return self.x * other.x + self.y * other.y

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

if __name__ == "__main__":
    p1 = Point()
    p1 += Point(0.1, -0.1)
    print(p1)
