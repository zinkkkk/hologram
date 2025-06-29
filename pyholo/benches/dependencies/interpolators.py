import numpy as np
from scipy.interpolate import Rbf


class RBFscipy:
    def __init__(self, x, y, epsilon=None):
        self.x = x
        self.y = y
        self.epsilon = np.sqrt(epsilon)
        self.rbf = Rbf(
            *[x[:, i] for i in range(np.shape(x)[1])],
            y,
            epsilon=self.epsilon,
            function=self.kernel,
            norm="sqeuclidean",
        )
        # Note the euclidean is used because the default gaussian is squaring

    def predict(self, x_new):
        return self.rbf(*[x_new[:, i] for i in range(np.shape(x_new)[1])])

    def kernel(self, r):
        r[r < 2.2204460492503131e-12] = 2.2204460492503131e-12
        return np.exp(-0.5 * r / self.epsilon**2)


class RBFnumpy:
    def __init__(self, x, y, sigma=None):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.A = self._rbf_kernel(self.x, self.x)
        self.coef_ = np.linalg.lstsq(self.A, self.y, rcond=None)[0]

    def _rbf_kernel(self, X, Y):
        dist = cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * dist / self.sigma**2)

    def predict(self, x_new):
        A_new = self._rbf_kernel(x_new, self.x)
        return np.dot(A_new, self.coef_)


class RBFnumpyNormalised:
    def __init__(self, x, y, sigma=None):
        # Ensure inputs are numpy arrays
        self.x_orig = np.asarray(x)
        self.y_orig = np.asarray(y)

        # Store original shape for output
        self.output_shape = self.y_orig.shape[1:] if self.y_orig.ndim > 1 else (1,)

        # Normalize input features
        self.x_mean = np.mean(self.x_orig, axis=0)
        self.x_std = (
            np.std(self.x_orig, axis=0) + 1e-10
        )  # Add small constant to avoid division by zero
        self.x = (self.x_orig - self.x_mean) / self.x_std

        # Normalize output features
        self.y_mean = np.mean(self.y_orig, axis=0)
        self.y_std = np.std(self.y_orig, axis=0) + 1e-10
        self.y = (self.y_orig - self.y_mean) / self.y_std

        # Set default sigma if not provided
        if sigma is None:
            # Heuristic: use average distance between points
            if len(self.x) > 1:
                distances = cdist(self.x, self.x, "sqeuclidean")
                sigma = np.median(distances[~np.eye(len(distances), dtype=bool)])
            else:
                sigma = 1.0
        self.sigma = sigma

        # Compute kernel matrix and solve for coefficients
        self.A = self._rbf_kernel(self.x, self.x)
        self.coef_ = np.linalg.lstsq(self.A, self.y, rcond=None)[0]

    def _rbf_kernel(self, X, Y):
        """Compute the RBF kernel between X and Y"""
        dist = cdist(X, Y, "sqeuclidean")
        return np.exp(-0.5 * dist / self.sigma**2)

    def predict(self, x_new):
        """Predict using the RBF model with denormalization"""
        x_new = np.asarray(x_new)
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)

        # Normalize input
        x_norm = (x_new - self.x_mean) / self.x_std

        # Compute predictions in normalized space
        A_new = self._rbf_kernel(x_norm, self.x)
        y_pred_norm = np.dot(A_new, self.coef_)

        # Denormalize output
        y_pred = y_pred_norm * self.y_std + self.y_mean

        # Reshape to match output dimensions
        if len(self.output_shape) > 1:
            y_pred = y_pred.reshape(-1, *self.output_shape)

        return y_pred.squeeze()


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
