import numpy as np
import matplotlib.pyplot as plt
from sidrpy.classes import Point
from sidrpy.utils import point_location_from_vec

if __name__ == "__main__":
    x, y = np.random.rand(2, 3)

    plt.arrow(x[1],  # Начальная координата по x
              y[1],  # Начальная координата по y
              x[2] - x[1],  # Длина стрелки по x
              y[2] - y[1],  # Длина стрелки по y
              width=0.001,
              color="tab:blue",
              overhang=0.2,  # форма стрелки
              head_width=0.025,
              length_includes_head=True)

    plt.scatter(x, y)
    plt.title(point_location_from_vec(*[Point(*coord) for coord in zip(x, y)]))

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.show()
