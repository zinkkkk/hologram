import numpy as np
from scipy.interpolate import Rbf
from typing import Tuple, Dict, Optional


def normalise_data(
    data: np.ndarray, stats: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    data = np.asarray(data)
    if stats is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        stats = {"mean": mean, "std": std}

    normalised = (data - stats["mean"]) / stats["std"]
    return normalised, stats


def denormalise_data(
    normalised_data: np.ndarray, stats: Dict[str, np.ndarray]
) -> np.ndarray:
    normalised_data = np.asarray(normalised_data)
    return normalised_data * stats["std"] + stats["mean"]


class RBFscipy:
    def __init__(self, x, y, kernel_name="gaussian", epsilon=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.kernel = (
            self.gaussian_kernel
            if kernel_name == "gaussian"
            else self.thin_plate_spline_kernel
        )

        # Ensure x is 2D (samples, features)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)

        # If y is 1D, reshape to (samples, 1)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        self.epsilon = np.sqrt(epsilon) if epsilon is not None else 1.0
        self.n_outputs = self.y.shape[1] if self.y.ndim > 1 else 1

        # Create one RBF model per output dimension
        self.rbfs = []
        for i in range(self.n_outputs):
            y_col = self.y[:, i] if self.n_outputs > 1 else self.y.ravel()
            rbf = Rbf(
                *[self.x[:, j] for j in range(self.x.shape[1])],
                y_col,
                epsilon=self.epsilon,
                function=self.kernel,
                norm="sqeuclidean",
            )
            self.rbfs.append(rbf)

    def predict(self, x_new):
        x_new = np.asarray(x_new)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)

        # Predict each output dimension separately
        predictions = []
        for rbf in self.rbfs:
            pred = rbf(*[x_new[:, j] for j in range(x_new.shape[1])])
            predictions.append(pred.reshape(-1, 1))

        # Stack predictions for each output dimension
        return (
            np.hstack(predictions) if len(predictions) > 1 else predictions[0].ravel()
        )

    def gaussian_kernel(self, r):
        r[r < 2.2204460492503131e-12] = 2.2204460492503131e-12
        return np.exp(-0.5 * r / self.epsilon**2)

    def thin_plate_spline_kernel(self, r):
        # Implementation matching the Rust version
        r = np.maximum(r, 2.2204460492503131e-12)  # Avoid log(0)
        return np.where(r > 0, r * np.log(np.sqrt(r)), 0.0)


class RBFnumpy:
    def __init__(self, x, y, kernel_name="gaussian", sigma=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.kernel = (
            self.gaussian_kernel
            if kernel_name == "gaussian"
            else self.thin_plate_spline_kernel
        )

        # Ensure x is 2D (samples, features)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)

        # If y is 1D, reshape to (samples, 1)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

        self.sigma = sigma
        self.kernel_name = kernel_name
        self.A = self._rbf_kernel(self.x, self.x)
        self.coef_ = np.linalg.lstsq(self.A, self.y, rcond=None)[0]

    def _rbf_kernel(self, X, Y):
        dist = cdist(X, Y, "sqeuclidean")
        return self.kernel(dist)

    def gaussian_kernel(self, r):
        return np.exp(-0.5 * r / self.sigma**2)

    def thin_plate_spline_kernel(self, r):
        r = np.sqrt(r + 1e-12)  # Add small constant to avoid log(0)
        return np.where(r > 0, r * r * np.log(r), 0.0)

    def predict(self, x_new):
        x_new = np.asarray(x_new)
        if x_new.ndim == 1:
            x_new = x_new.reshape(-1, 1)
            
        A_new = self._rbf_kernel(x_new, self.x)
        result = np.dot(A_new, self.coef_)
        
        # Match SciPy's behavior: return 1D array for single output, 2D otherwise
        if result.shape[1] == 1:
            return result.ravel()
        return result


def cdist(XA, XB, metric="euclidean"):
    """
    Computes the distance between each pair of vectors in XA and XB using the specified metric.
    XA and XB should be 2D arrays, where each row represents a vector.
    """
    m = XA.shape[0]
    n = XB.shape[0]
    d = XA.shape[1]
    assert d == XB.shape[1], "Dimensionality of XA and XB must match."
    dist = np.zeros((m, n))
    if metric == "euclidean":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sqrt(np.sum((XA[i] - XB[j]) ** 2))
    elif metric == "cityblock":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sum(np.abs(XA[i] - XB[j]))
    elif metric == "sqeuclidean":
        for i in range(m):
            for j in range(n):
                dist[i, j] = np.sum((XA[i] - XB[j]) ** 2)
    elif metric == "cosine":
        for i in range(m):
            for j in range(n):
                dist[i, j] = 1 - np.dot(XA[i], XB[j]) / (
                    np.linalg.norm(XA[i]) * np.linalg.norm(XB[j])
                )
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return dist
