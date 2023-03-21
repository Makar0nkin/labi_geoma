import numpy as np
import matplotlib.pyplot as plt
from sidrpy.utils import are_two_lines_intersect
from sidrpy.classes import Point

if __name__ == "__main__":
    x = np.random.rand(4)
    y = np.random.rand(4)
    # x = [0.1, 0.3, 0.5, 0.7]
    # y = [0.5, 0.5, 0.5, 0.5]

    plt.plot(x[0:2], y[0:2])
    plt.plot(x[2:4], y[2:4])

    if are_two_lines_intersect(*[Point(*coord) for coord in zip(x, y)]):
        plt.title("Пересекаются")
    else:
        plt.title("Не пересекаются")

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.show()
