import sys

# Add the parent directory to sys.path to find the dependencies module for pytest
sys.path.insert(0, "../.venv/lib/python3.12/site-packages/pyholo/")
sys.path.insert(0, "../.venv/lib/python3.12/site-packages/")
sys.path.insert(0, ".")

import numpy as np
from random import uniform
import dependencies.interpolators as py
import pyholo as holo

dimension = 100
bounds = (-5.0, 5.0)
n_train = 10
n_test = 5


def blackbox(pts):
    return 10.0 * dimension + np.sum(
        np.power(pts, 2) + 10.0 * np.cos(2.0 * np.pi * pts)
    )


x_train = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(n_train)]
)
y_train = np.array([blackbox(x) for x in x_train])

training_data = (x_train, y_train)

x_new = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(n_test)]
)


def pyholo_rbf_model(training_data, new_data):
    newrbf = holo.Rbf(training_data[0], training_data[1], "gaussian", 1.0)
    pred1 = newrbf.predict(new_data)


def scipy_model(training_data, new_data):
    newrbf = py.RBFscipy(training_data[0], training_data[1], "gaussian", 1.0)
    pred1 = newrbf.predict(new_data)


def numpy_model(training_data, new_data):
    newrbf = py.RBFnumpy(training_data[0], training_data[1], "gaussian", 1.0)
    pred1 = newrbf.predict(new_data)


def test_scipy(benchmark):
    benchmark(scipy_model, training_data, x_new)


def test_pyholo(benchmark):
    benchmark(pyholo_rbf_model, training_data, x_new)


def test_numpy(benchmark):
    benchmark(numpy_model, training_data, x_new)
