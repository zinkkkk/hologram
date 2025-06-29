from random import uniform
import numpy as np

import dependencies.interpolators as py
import pyholo as holo

dimension = 2


def blackbox(pts):
    return 10.0 * dimension + np.sum(
        np.power(pts, 2) + 10.0 * np.cos(2.0 * np.pi * pts)
    )


bounds = (-5.0, 5.0)

x_train = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(10)]
)
y_train = np.array([blackbox(x) for x in x_train])

training_data = (x_train, y_train)

x_new = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(50)]
)


def pyholo_rbf_model(training_data, new_data):
    # Train the rbf model
    newrbf = holo.Rbf(training_data[0], training_data[1], "gaussian", 1.0)
    # newrbf = fl.Rbf(training_data[0], training_data[1], "gaussian", 1.0)
    pred1 = newrbf.predict(new_data[:5])
    print(np.array(pred1))


def scipy_model(training_data, new_data):
    # Train the rbf model
    newrbf = py.RBFscipy(training_data[0], training_data[1], 1.0)
    pred1 = newrbf.predict(new_data[:5])
    print(pred1)


def numpy_model(training_data, new_data):
    # Train the rbf model
    newrbf = py.RBFnumpyNormalised(training_data[0], training_data[1], 1.0)
    pred1 = newrbf.predict(new_data[:5])
    print(pred1)


def test_scipy(benchmark):
    benchmark(scipy_model, training_data, x_new)


def test_pyholo(benchmark):
    benchmark(pyholo_rbf_model, training_data, x_new)


def test_numpy(benchmark):
    benchmark(numpy_model, training_data, x_new)


print("Scipy:")
scipy_model(training_data, x_new)

print("Pyholo:")
pyholo_rbf_model(training_data, x_new)

print("Numpy:")
numpy_model(training_data, x_new)
