#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pywt
from numpy import linalg as la



def cgd_solver(X, y, Gamma, lambda1, lambda2, eps = 1e-4, max_it = 50000):
    m, p = Gamma.shape
    X_til, y_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma)), np.concatenate((y, np.zeros(m)))
    X_til_pinv = la.pinv(X_til)

    y_v = X_til @ X_til_pinv @ y_til
    Gamma_v = Gamma @ X_til_pinv

    Q = Gamma_v @ Gamma_v.T
    b = Gamma_v @ y_v

    u = np.zeros(m)
    n_iter = 0
    prev_u = 0 # For stopping criteria
    while True:
        n_iter += 1
        if n_iter >= max_it:
            #raise ValueError("Iterations exceed max_it")
            print("Iterations exceed max_it")
            return X_til_pinv @ (y_v - Gamma_v.T @ u)
        for i in range(m):
            if Q[i, i] > 1e-4:
                t = 1/Q[i,i] * (b[i] - np.dot(np.delete(Q[i], i), np.delete(u, i)))
            else:
                t = 0

            u[i] = np.sign(t) * min(np.abs(t), lambda1)   #there should be better truncation methods

        if la.norm(u - prev_u) <= eps:
            break

        prev_u = np.copy(u)   # Recall array is similar to list

    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta


# ### Generate stair shape toy example

# In[6]:
