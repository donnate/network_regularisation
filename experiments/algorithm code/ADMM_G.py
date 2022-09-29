#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pywt
from numpy import linalg as la
from scipy.linalg import cho_factor, cho_solve


# In[2]:


def beta_update(Gamma, v, nu, rho, Cho, M2):  #Cho is a tuple of the 2 Cholesky factorization components
     return cho_solve(Cho, M2 + rho* Gamma.T@(v + nu/rho))
    
    
def v_update(Gamma, beta, v, nu, lambda1, lambda2, rho):
    w = Gamma @ beta - 1/rho * nu
    
    new_v = 1/(1 + 2*lambda2/rho) * np.sign(w) * pywt.threshold(np.abs(w), lambda1/rho, 'soft' )
    r_d = rho * Gamma.T @(v-new_v)
    return new_v, r_d


def nu_update(Gamma, beta, v, nu, rho):
    r_p = v - Gamma @ beta
    return nu + rho * r_p, r_p


# In[3]:


def Gauss_ADMM(X, y, Gamma, lambda1, lambda2, rho = 1, eps = 1e-3, max_it = 50000):
    m, p = Gamma.shape
    
    L = Gamma.T @ Gamma
    M1, M2 = X.T@X + rho*L, X.T@y
    ## if la.det(M1) == 0:
        ### raise ValueError("Matrix is singular; update for beta isn't unique")
        
    M1_Cho = cho_factor(M1)
    
    n_iter = 0
    v, nu = np.zeros(m), np.zeros(m)
    while True:
        n_iter += 1
        if n_iter >= max_it:
            # raise ValueError("Iterations exceed max_it")
            print("Iterations exceed max_it")
            return beta
        beta = beta_update(Gamma, v, nu, rho, M1_Cho, M2)
        v, r_d = v_update(Gamma, beta, v, nu, lambda1, lambda2, rho)
        nu, r_p = nu_update(Gamma, beta, v, nu, rho)
        if la.norm(r_d) <= eps and la.norm(r_p) <= eps:
            break
    return beta
        


# ### Toy example: Scaling $n = p$
# 
# #### Covariance matrix $\Psi$ is blockwise fully correlated. True $\beta^*$ is piece-wise constant correspondingly (stair-shape). $\Gamma$ is from the 1-D linear chain graph. 
# 
# #### Set $x_i \sim N(0, \Psi), \lambda_1 = \lambda_2 = 0.2, \rho = 0.1, \epsilon = 1e-4$ 
# #### Note: we can now set $\rho = 0.1$ (and even smaller), we can't do it for objective function with $1/n$. This version is better.

# In[4]:






    


