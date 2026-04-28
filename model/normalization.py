import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        eps = 1e-5
        x_norm = (x - np.mean(x)) / np.sqrt(np.var(x) + eps)
        out = gamma * x_norm + beta
        return np.round(out, 5)
