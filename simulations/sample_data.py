import numpy as np


def gaussian_sample(n, p, beta_star, Psi, sigma, set_seed=1):
    np.random.seed(set_seed)
    print(beta_star.shape, Psi.shape)
    X = np.random.multivariate_normal(mean = np.zeros(p),
                                      cov=Psi, size=n)
    y = X.dot(beta_star)+ sigma * np.random.normal(size=n)
    return X, y
