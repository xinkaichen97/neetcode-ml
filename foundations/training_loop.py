import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            # forward pass
            y_hat = X @ w + b
            error = y_hat - y

            # calculate gradients
            dw = 2.0 * X.T @ error / n
            db = 2.0 * np.sum(error) / n

            # update weights
            w -= lr * dw
            b -= lr * db

        return np.round(w, 5), round(float(b), 5)
