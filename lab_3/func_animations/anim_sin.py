import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

plt.style.use('dark_background')

"""
        In lines(7–9), we simply create a figure window with a single axis in the figure. 
        Then we create our empty line object which is essentially the one to be modified 
        in the animation. The line object will be populated with data later.
        
        In lines(11–13), we create the init function that will make the animation happen. 
        The init function initializes the data and also sets the axis limits.
        
        In lines(14–18), we finally define the animation function which takes in the frame number(i)
        as the parameter and creates a sine wave(or any other animation) which a shift depending
        upon the value of i. This function here returns a tuple of the plot objects 
        which have been modified which tells the animation framework what parts of the plot 
        should be animated.
        
        In line 20, we create the actual animation object. The blit parameter ensures 
        that only those pieces of the plot are re-drawn which have been changed.
"""

fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,


anim = FuncAnimation(fig, animate,
                     init_func=init, frames=200, interval=20, blit=True)

print(animation.writers.list())
anim.save('sine_wave.gif', writer='ffmpeg')
