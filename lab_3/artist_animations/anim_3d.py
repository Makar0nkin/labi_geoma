import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
import numpy as np

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

x_grid, y_grid = np.meshgrid(x, y)

phasa = np.arange(0, 2*np.pi, 0.1)
frames = []

for p in phasa:
    z_grid = np.sin(x_grid + p) * np.sin(y_grid) / (x_grid * y_grid)
    line = ax.plot_surface(x_grid, y_grid, z_grid, color="tab:blue")
    frames.append([line])

print(frames)
ani = ArtistAnimation(
    fig,
    frames,
    interval=30,
    blit=True,
)
ani.save(f"3d_sample.gif", dpi=300, writer=PillowWriter(fps=30))