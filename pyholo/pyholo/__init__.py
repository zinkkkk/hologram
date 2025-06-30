from .pyholo import Rbf as RustRbf

__doc__ = pyholo.__doc__
if hasattr(pyholo, "__all__"):
    __all__ = pyholo.__all__


def _flatten_if_singleton_2d(array):
    if (
        isinstance(array, list)
        and len(array) > 0
        and isinstance(array[0], list)
        and all(isinstance(sub, list) and len(sub) == 1 for sub in array)
    ):
        return [sub[0] for sub in array]
    return array


class Rbf:
    """
    Radial basis function interpolator.

    A flexible interpolator that supports scalar and vector input/output using various kernel functions.

    Examples
    --------
    >>> import pyholo as holo
    >>> x_train = np.array([0.000, 512.000, 1182.490, 1911.273, 2788.547, 4227.750, 6481.706, 9609.367])
    >>> y_train = np.array([
        [6950.000, 0.000, 0.000],
        [5930.999, 4386.866, 974.859],
        [2498.780, 8537.536, 1897.230],
        [-1993.234, 10819.437, 2404.319],
        [-7051.251, 11465.688, 2547.931],
        [-13549.361, 9748.529, 2166.340],
        [-19093.612, 3887.659, 863.924],
        [-18104.032, -5716.279, -1270.284]
    ])
    >>> rbf = holo.Rbf(x_train, y_train, kernel="gaussian", epsilon=1.0)
    >>> y_pred = rbf.predict([[800.0, 900.0, 1000.0]])
    """

    def __init__(self, x, y, kernel=None, epsilon=None):
        """
        Instantiate a radial basis function (RBF) interpolator.

        Parameters
        ----------
        x : list[float] or list[list[float]] or np.ndarray
            The input data points. Can be 1D (e.g., [1.0, 2.0, 3.0]) or 2D
            (e.g., [[1.0], [2.0], [3.0]]). 2D arrays with singleton inner lists
            will be flattened automatically.
        y : list[float] or list[list[float]] or np.ndarray
            The corresponding output values. Can be 1D or 2D. Automatically flattened
            if 2D with shape (n, 1).
        kernel : str or None, optional
            The kernel function to use. Supported values include:
            "gaussian", "linear", "cubic", "multiquadric",
            "inverse_multiquadratic", and "thin_plate_spline".
            Defaults to "gaussian".
        epsilon : float or None, optional
            Bandwidth parameter for the kernel function. Controls the spread of the RBF.
            Defaults to 1.0 if not specified.
        """

        # Flatten if necessary
        x = _flatten_if_singleton_2d(x)
        y = _flatten_if_singleton_2d(y)

        # Delegate to the Rust implementation
        self._inner = RustRbf(x, y, kernel, epsilon)

    def predict(self, x_new):
        """Predict the output values for the given input data points.

        Parameters
        ----------
        x_new : list[float] | list[list[float]] | np.ndarray
            The new data points.

        Returns
        -------
        list[float] | list[list[float]] | np.ndarray
            The predicted output data points.
        """
        x_new = _flatten_if_singleton_2d(x_new)
        return self._inner.predict(x_new)

    def __repr__(self):
        """Return a string representation of the object."""
        return repr(self._inner)
