import sys

# Add the parent directory to sys.path to find the dependencies module for pytest
sys.path.insert(0, "../.venv/lib/python3.12/site-packages/pyholo/")
sys.path.insert(0, "../.venv/lib/python3.12/site-packages/")
sys.path.insert(0, ".")

import numpy as np
import dependencies.interpolators as py
import pyholo as holo

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

training_data = (x_train, y_train)

x_new = np.linspace(x_train[0], x_train[-1], 200)


def pyholo_model(training_data, new_data):
    newrbf = holo.Rbf(training_data[0], training_data[1], "thin_plate_spline", 1.0)
    pred1 = newrbf.predict(new_data)


def scipy_model(training_data, new_data):
    newrbf = py.RBFscipy(training_data[0], training_data[1], "thin_plate_spline", 1.0)
    pred1 = newrbf.predict(new_data)


def numpy_model(training_data, new_data):
    newrbf = py.RBFnumpy(training_data[0], training_data[1], "thin_plate_spline", 1.0)
    pred1 = newrbf.predict(new_data)


def test_pyholo(benchmark):
    benchmark(pyholo_model, training_data, x_new)


def test_scipy(benchmark):
    benchmark(scipy_model, training_data, x_new)
