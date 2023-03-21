import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure()
# creating a subplot
ax1 = fig.add_subplot(1, 1, 1)


def animate(i):
    print(i)
    xs = np.random.rand(10)
    ys = np.random.rand(10)
    ax1.clear()
    ax1.plot(xs, ys)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live graph with matplotlib')


ani = animation.FuncAnimation(fig, animate)
plt.show()