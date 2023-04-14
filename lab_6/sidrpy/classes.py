from dataclasses import dataclass
import math



@dataclass
class Point:
    x: float = 0
    y: float = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        self.x += other.x_arr
        self.y += other.y
        return self

    def __sub__(self, other):
        self.x -= other.x_arr
        self.y -= other.y
        return self

    def __mul__(self, other):
        return self.x * other.x_arr + self.y * other.y

    def __eq__(self, other):
        return self.x == other.x_arr and self.y == other.y

@dataclass
class Vector(Point):
    def __init__(self, p1: Point, p2: Point):
        self.x = p2.x - p1.x
        self.y = p2.y - p1.y

    def get_length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __mul__(self, other):
        return self.x * other.x_arr + self.y * other.y

if __name__ == "__main__":
    p1 = Point()
    p1 += Point(0.1, -0.1)
    print(p1)
