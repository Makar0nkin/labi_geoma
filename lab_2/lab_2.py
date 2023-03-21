import numpy as np
import matplotlib.pyplot as plt
from sidrpy.polygon_utils import point_location_from_simple_polygon
from sidrpy.classes import Point

# координаты многоугольника и точки

x0, y0 = np.random.random(2) * 12
p0 = Point(x0, y0)
x = [2, 10, 3, 2]
y = [3, 2, 5, 10]

plt.title(point_location_from_simple_polygon(x, y, p0))

# построение многоугольника
plt.plot([*x, x[0]], [*y, y[0]])
plt.scatter(x0, y0)
plt.show()
