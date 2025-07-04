import numpy as np
import matplotlib.pyplot as plt
import dependencies.interpolators as py
import pyholo as holo
import time

def blackbox(pts):
    return 10.0 + np.sum(
        np.power(pts, 2) + 10.0 * np.cos(2.0 * np.pi * pts)
    )

bounds = (-5.0, 5.0)

x_train = np.linspace(bounds[0], bounds[1], 50)
y_train = np.array([blackbox(x) for x in x_train])

x_new = np.linspace(bounds[0], bounds[1], 200)

# Get predictions from each model
start_time = time.time()
scipy_pred = py.RBFscipy(x_train, y_train, "gaussian", 1.0).predict(x_new)
end_time = time.time()
print(f"Scipy time taken: {end_time - start_time}")

start_time = time.time()
pyholo_pred = holo.Rbf(x_train, y_train, "gaussian", 1.0).predict(x_new)
end_time = time.time()
print(f"PyHolo time taken: {end_time - start_time}")

# start_time = time.time()
# numpy_pred = py.RBFnumpy(x_train, y_train, "gaussian", 1.0).predict(x_new)
# end_time = time.time()
# print(f"Numpy time taken: {end_time - start_time}")

# Create the plot
plt.figure(figsize=(12, 8))

# Plot true function
x_true = np.linspace(bounds[0], bounds[1], 500)
y_true = np.array([blackbox(x) for x in x_true])
plt.plot(x_true, y_true, 'k-', linewidth=2, label='True Function', alpha=0.5)

# Plot predictions
plt.plot(x_new, scipy_pred, 'b-.', linewidth=2, label='Scipy RBF')
plt.plot(x_new, pyholo_pred, 'r--', linewidth=2, label='Hologram RBF')
# plt.plot(x_new, numpy_pred, 'g-.', linewidth=2, label='Numpy RBF')

# Add training points
plt.scatter(x_train, y_train, color='black', s=100, zorder=5, 
           label='Training Points', edgecolors='white', linewidth=1.5)

# Add labels and title
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output (y)', fontsize=12)
plt.title('RBF Interpolation Comparison (1D Input, 1D Output)', fontsize=14)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.show()
