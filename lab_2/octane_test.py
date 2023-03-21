import numpy as np
import matplotlib.pyplot as plt
from sidrpy.classes import Point
from sidrpy.polygon_utils import is_polygon_simple, point_location_from_simple_polygon_octane

n = 6
x, y = np.random.rand(2, n)
while not is_polygon_simple(x, y):
    x, y = np.random.rand(2, n)

x0, y0 = np.random.random(2)
p0 = Point(x0, y0)

plt.title(point_location_from_simple_polygon_octane(x, y, p0))
plt.plot([*x, x[0]], [*y, y[0]])
plt.scatter(x0, y0)
plt.show()
