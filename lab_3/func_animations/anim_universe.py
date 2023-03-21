import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fps = 30
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r*')

def init():
   ax.set_xlim(0, 100)
   ax.set_ylim(-1, 1)
   return ln,
def animate(frame):
   xdata.append(frame)
   ydata.append(np.sin(frame))
   ln.set_data(xdata, ydata)
   return ln,

ani = FuncAnimation(fig, animate, init_func=init, blit=True, frames=100)
ani.save(f"simple_animation_{fps}fps.gif", dpi=300, writer=PillowWriter(fps=fps))
# plt.show()