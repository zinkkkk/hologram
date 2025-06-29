import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

def plot_earth(ax, radius, resolution=100):
    p = np.linspace(0, np.pi, resolution)
    t = np.linspace(0, 2 * np.pi, resolution)
    P, T = np.meshgrid(p, t)

    x = radius * np.cos(T) * np.sin(P)
    y = radius * np.sin(T) * np.sin(P)
    z = radius * np.cos(P)

    ax.plot_surface(x, y, z, cmap='winter', alpha=0.25, edgecolor='none')

def plot_orientation(ax, size):
    l = size
    x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    u, v, w = [[-l, 0, 0], [0, l, 0], [0, 0, l]]
    ax.quiver(x, y, z, u, v, w, color='w')

def set_axes(ax, max_val):
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('auto')

    ax.set_xlabel('km')
    ax.set_ylabel('km')
    ax.set_zlabel('km')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

def plot_positions(ax, positions_x, positions_y, positions_z, color='r', label='Line'):
    ax.plot3D(positions_x, positions_y, positions_z, color, label=label)

