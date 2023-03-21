import numpy as np
import matplotlib.pyplot as plt
from sidrpy.utils import is_polygon_simple

if __name__ == "__main__":
    n = np.random.randint(4, 5 + 1)
    x, y = np.random.rand(2, n)

    plt.title("Простой" if is_polygon_simple(x, y) else "Не простой")

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.plot([*x, x[0]], [*y, y[0]])
    plt.scatter(x, y)
    plt.show()
