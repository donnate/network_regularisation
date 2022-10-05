#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy import linalg as la
### Notice, use scipy.sparse.diags can cause dimension mismatch problems. Use only NumPy instead.
from scipy.linalg import cho_factor, cho_solve


def directions(D, u, y, lambda1, mu1, mu2, t):
    m = u.shape[0]
    f1 = u - lambda1 * np.ones(m)
    f2 = - u - lambda1 * np.ones(m)

    J1_inv = np.diag(mu1/f1)
    J2_inv = np.diag(mu2/f2)

    # Directions
    ##Solve A(d_u) = w

    M = D @ D.T
    A_Cho = cho_factor(M - J1_inv - J2_inv)
    w = -(M @ u - D @ y - 1/t* 1/f1 + 1/t * 1/f2)
    d_u = cho_solve(A_Cho, w)


    d_mu1 = -(mu1 + 1/t * 1/f1 + J1_inv @ d_u)
    d_mu2 = -(mu2 + 1/t * 1/f2 - J2_inv @ d_u)


    return d_u, d_mu1, d_mu2


def line_search(D, u, y, lambda1, mu1, mu2, t, d_u, d_mu1, d_mu2, a = 0.1, b = 0.7):
    m = u.shape[0]
    f1 = (u - lambda1 * np.ones(m)).reshape((m,1))
    f2 = (- u - lambda1 * np.ones(m)).reshape((m,1))

    a0 = -mu1/d_mu1

    a1 = np.concatenate((a0[a0 >= 0],np.ones(1)))             #avoid empty set
    b0 = -mu2/d_mu2
    b1 = np.concatenate((b0[b0 >= 0],np.ones(1)))

    s_max = min(1, min(a1), min(b1))
    s = 0.99 * s_max

    while np.any(f1 + s* d_u >=0) and np.any(f2 - s* d_u >=0):
        s = b*s

    r_t_0 = residuals(D, u, y, lambda1, mu1, mu2, t)

    while residuals(D, u+s*d_u, y, lambda1, mu1+s*d_mu1, mu2+s*d_mu2, t) > (1-a*s) * r_t_0:
        s = b *s

    return s


def residuals(D, u, y, lambda1, mu1, mu2, t):
    m = u.shape[0]
    f1 = u - lambda1 * np.ones(m)
    f2 = - u - lambda1 * np.ones(m)

    # residuals
    M = D @ D.T
    r1 = M @ u - D @ y + mu1 - mu2
    r2 = -np.diag(mu1) @ f1 - 1/t * np.ones(m)
    r3 = -np.diag(mu2) @ f2 - 1/t * np.ones(m)

    r_t = la.norm(np.concatenate((r1, r2, r3)))

    return r_t


def s_gap(u, lambda1, mu1, mu2):
    m = u.shape[0]
    f1 = u - lambda1 * np.ones(m)
    f2 = - u - lambda1 * np.ones(m)
    return -f1 @ mu1 - f2 @ mu2


def ip_solver(X,y, Gamma, lambda1, lambda2, mu = 1.5, eps = 1e-4, max_it = 10000):
    m, p = Gamma.shape
    X_til, y_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma)), np.concatenate((y, np.zeros(m)))
    X_til_pinv = la.pinv(X_til)

    y_v = X_til @ X_til_pinv @ y_til
    Gamma_v = Gamma @ X_til_pinv

    u = np.ones(m)
    mu1 = 10*np.ones(m)
    mu2 = 10*np.ones(m)

    t = 2*m*mu/s_gap(u, lambda1, mu1, mu2)

    n_iter = 0

    while True:
        n_iter += 1
        if n_iter >= max_it:
            #raise ValueError("Iterations exceed max_it")
            print("Iterations exceed max_it")
            return beta

        d_u, d_mu1, d_mu2 = directions(Gamma_v, u, y_v, lambda1, mu1, mu2, t)
        s = line_search(Gamma_v, u, y_v, lambda1, mu1, mu2, t, d_u, d_mu1, d_mu2)
        u += s*d_u
        mu1 += s*d_mu1
        mu2 += s*d_mu2


        r_t = residuals(Gamma_v, u, y_v, lambda1, mu1, mu2, t)
        eta = s_gap(u, lambda1, mu1, mu2)

        t = 2*m*mu/eta   # 2m since we have mu1 and mu2 of total 2m variables

        if r_t <= eps and eta <= eps:
            break


    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta
