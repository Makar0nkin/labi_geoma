from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Point:
    x: float = 0
    y: float = 0


@dataclass
class Vector(Point):

    def __init__(self, p1: Point, p2: Point):
        self.x = p2.x - p1.x
        self.y = p2.y - p1.y


    def get_length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def __mul__(self, other):
        return self.x * other.x_arr + self.y * other.y

