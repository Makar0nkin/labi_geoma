import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
ims = []
for t in np.arange(15):
    # plt.clear()
    plt.scatter(x + t, y + t ** 2)
    ims.append(plt.plot(x + t, y + t ** 2))
    ims.append(plt.plot([10, 20, 30], [20, 40, 60]))

print(ims)
im_ani = animation.ArtistAnimation(fig, ims, interval=50)
im_ani.save('cho-to.gif', writer='ffmpeg')
