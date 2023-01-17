
import numpy as np
import matplotlib.pyplot as plt
from estimators import *
import scipy
import time
import networkx as nx
from sklearn.model_selection import GridSearchCV
import random
import numpy.linalg as la
import pywt


def smooth_stair(slope_length, slope_height, step_length, start_value, n_repeat): #total size = n_repeat * (slope_l + step_l) - step_l
    part = [slope_height/2 + slope_height/2 * np.sin(j * np.pi/slope_length) for j in np.arange(-slope_length/2, slope_length/2, 1)] + step_length * [slope_height]
    b = []
    for i in range(n_repeat):
        b += [x + slope_height*i for x in part]
    return np.array([x + start_value for x in b][slope_length: ])

def chain_incidence(p):
    return scipy.sparse.diags([np.ones(p), -np.ones(p-1)], [0, 1]).toarray()[0:p-1,:]
def grid_incidence(a): #a is side len
    G = nx.grid_graph(dim = (a, a))
    return np.asarray(nx.incidence_matrix(G, oriented = True).T.todense())

def toeplitz(a, p):
    return np.exp(np.log(a)*np.abs(np.subtract.outer(range(p), range(p))))

def gauss_sample(n, p, beta_star, Psi, sigma, set_seed = 1):
    random.seed(set_seed)
    X = np.random.multivariate_normal(mean = np.zeros(p), cov=Psi, size=n)
    y = X.dot(beta_star) + sigma * np.random.normal(size=n)
    return X, y


def cor_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    W = np.sqrt(np.diag(np.diagonal(C)))
    C_1 = la.inv(W)@C@la.inv(W)
    return C_1



## Caution: may have bugs
def smooth_2d(side_len, disk_radi, period, height, start_value):  # for symmetry, side_len should be odd
                                                                  # set period = 1 for piece-wise constant case                                                                                                      
    if side_len % 2 == 0:
        left_end = -(side_len//2) + 1
    else:
        left_end = -(side_len//2)
    x, y = np.meshgrid(np.linspace(left_end, side_len//2, side_len),np.linspace(left_end, side_len//2, side_len))  
    ds = np.sqrt(x**2+y**2)
    ds = pywt.threshold(ds, disk_radi, 'soft')
    ds[abs(ds) > period] = -period
    ds = np.array([x + period/2 for x in ds])
    part = np.sin(np.pi*ds/period)
    return np.array([height/2 * x + start_value + height/2 for x in part])


