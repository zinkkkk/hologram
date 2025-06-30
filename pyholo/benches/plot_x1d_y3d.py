from dependencies.plot_utils import plot_earth, plot_orientation, set_axes
import pyholo as holo
import dependencies.interpolators as py
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")


x_train = np.array(
    [
        0.000,
        512.000,
        1182.490,
        1911.273,
        2788.547,
        4227.750,
        6481.706,
        9609.367,
        11773.210,
        13188.649,
        14400.000,
    ]
)

y_train = np.array(
    [
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
    ]
)

# Normalize the training data
x_train_norm, x_stats = py.normalise_data(x_train.reshape(-1, 1))
y_train_norm, y_stats = py.normalise_data(y_train)

# Prepare normalized training data
training_data_norm = (x_train_norm, y_train_norm)

# Generate test points and normalize them
x_new = np.linspace(x_train[0], x_train[-1], 200).reshape(-1, 1)
x_new_norm = py.normalise_data(x_new, x_stats)[0]

# Make predictions with each model using normalized data
pyholo_pred_norm = np.array(
    holo.Rbf(
        x_train_norm, 
        y_train_norm, 
        "gaussian", 
        1.0
    ).predict(x_new_norm)
)
pyholo_pred = py.denormalise_data(pyholo_pred_norm, y_stats)

scipy_pred_norm = py.RBFscipy(
    x_train_norm, 
    y_train_norm, 
    "gaussian", 
    1.0
).predict(x_new_norm)
scipy_pred = py.denormalise_data(scipy_pred_norm, y_stats)

numpy_pred_norm = py.RBFnumpy(
    x_train_norm, 
    y_train_norm, 
    "gaussian", 
    1.0
).predict(x_new_norm)
numpy_pred = py.denormalise_data(numpy_pred_norm, y_stats)

radius_earth = 6378.12

# Create a new figure and a 3D axis
fig = plt.figure(figsize=(13, 11))

orientation_size = 10000  # length of the orientation vectors in km
max_val = 25000  # for setting axis limits

ax = fig.add_subplot(111, projection="3d")
ax.grid(visible=False)
ax.set_axis_off()

# Plot the Earth
plot_earth(ax, radius_earth, 50)

# Plot the orientation vectors
plot_orientation(ax, orientation_size)

# Integrator
ax.plot3D(
    y_train[:, 0], y_train[:, 1], y_train[:, 2], "yo", markersize=10, label="Integrator"
)

# PyHolo
ax.plot3D(
    pyholo_pred[:, 0],
    pyholo_pred[:, 1],
    pyholo_pred[:, 2],
    "r",
    markersize=10,
    label="PyHolo",
)

# Scipy
ax.plot3D(
    scipy_pred[:, 0],
    scipy_pred[:, 1],
    scipy_pred[:, 2],
    "b",
    markersize=10,
    label="Scipy",
)

# Numpy
ax.plot3D(
    numpy_pred[:, 0],
    numpy_pred[:, 1],
    numpy_pred[:, 2],
    "c",
    markersize=10,
    label="Numpy",
)

# Set the axes
set_axes(ax, max_val)

# Adjust the view limits to ensure the whole plot is visible
ax.set_box_aspect([-1, -1, 1])

# Disable clipping
for spine in ax.spines.values():
    spine.set_clip_on(False)

# Add legend to the plot
ax.legend(loc="upper right", fontsize="large")

# Adjust plot margins to ensure everything is visible
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Show the plot
plt.show()
