import numpy as np


def toeplitz_covariance(a, p):
    return np.exp(-np.log(a) * np.abs(np.subtract.outer(range(p),
                    range(p))))
