from dependencies.plot_utils import plot_earth, plot_orientation, set_axes
import pyholo as holo
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("dark_background")


x_train = np.array([
    [0.000],
    [512.000],
    [1182.490],
    [1911.273],
    [2788.547],
    [4227.750],
    [6481.706],
    [9609.367],
    [11773.210],
    [13188.649],
    [14400.000],
])

y_train = np.array([
    [6950.000, 0.000, 0.000],
    [5930.999, 4386.866, 974.859],
    [2498.780, 8537.536, 1897.230],
    [-1993.234, 10819.437, 2404.319],
    [-7051.251, 11465.688, 2547.931],
    [-13549.361, 9748.529, 2166.340],
    [-19093.612, 3887.659, 863.924],
    [-18104.032, -5716.279, -1270.284],
    [-11425.229, -10664.783, -2369.952],
    [-4206.775, -11312.308, -2513.846],
    [3167.219, -7987.115, -1774.914],
])

training_data = (x_train, y_train)

x_new = np.linspace(x_train[0], x_train[-1], 200).reshape(-1, 1)

newrbf = holo.Rbf(training_data[0], training_data[1], "thin_plate_spline", 1.0)
# newrbf = py.RBFnumpyNormalised(training_data[0], training_data[1], 1.0)
pred1 = np.array(newrbf.predict(x_new))

print(pred1)

radius_earth = 6378.12

# Create a new figure and a 3D axis
fig = plt.figure(figsize=(13,11))

orientation_size = 10000  # length of the orientation vectors in km
max_val = 25000  # for setting axis limits

ax = fig.add_subplot(111, projection='3d')
ax.grid(visible=False)
ax.set_axis_off()

# Plot the Earth
plot_earth(ax, radius_earth, 50)

# Plot the orientation vectors
plot_orientation(ax, orientation_size)

# Integrator
positions_x = y_train[:, 0]
positions_y = y_train[:, 1]
positions_z = y_train[:, 2]
ax.plot3D(positions_x, positions_y, positions_z, 'yo', markersize=10, label='Integrator')

# Numpy
positions_x = pred1[:, 0]
positions_y = pred1[:, 1]
positions_z = pred1[:, 2]
ax.plot3D(positions_x, positions_y, positions_z, 'c', markersize=10, label='Interpolation')

# Set the axes
set_axes(ax, max_val)

# Adjust the view limits to ensure the whole plot is visible
ax.set_box_aspect([-1, -1, 1])

# Disable clipping
for spine in ax.spines.values():
    spine.set_clip_on(False)

# Add legend to the plot
ax.legend(loc='upper right', fontsize='large')

# Adjust plot margins to ensure everything is visible
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Show the plot
plt.show()