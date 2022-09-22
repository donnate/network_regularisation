
import numpy as np
import matplotlib.pyplot as plt
from skest import *
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


def comp_incidence(p):
    return np.asarray((nx.incidence_matrix(nx.complete_graph(p), oriented = True)).T.todense())
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

gridlasso = {'l1': [0,0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100], 'l2':[0]}

grid1 = {'l1': [0,0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20], 
'l2': [0,0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20]}
#grid1 = {'l1': [0,0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100], 'l2': [0,0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100]}
grid1small = {'l1': [0,0.1, 1, 10, 20, 50, 100, 200], 
'l2': [0,0.1, 1, 10, 20, 50, 100, 200]}
#gridGTV = {'l1': [0, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75], 
          #'l2': [0, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75],
          #'l3': [0]} #for 1 D
gridGTV = {'l1': [0, 0.001, 0.01,0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20], # for 2D
           'l2': [0, 0.001, 0.01,0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20],
          'l3': [0]} #save time
gridGTVsmall = {'l1': [0, 0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 25], 
          'l2': [0, 0.1, 0.5, 1, 2, 5, 3,  7.5, 10, 15, 25],
          'l3': [0, 0.1, 0.25, 0.5, 1, 2, 5]}  #1D
#gridGTVsmall = {'l1': [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5], 
           #'l2': [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
          #'l3': [0]} #2D
gridlogit  = {'l1': [0, 0.005, 0.01, 0.1, 0.2, 0.5], 
           'l2': [0, 0.005, 0.01, 0.1, 0.2, 0.5]}
gridlarge = {'l1': [0,0.1, 0.5, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 70, 80, 100, 125, 150], 
           'l2': [0,0.1, 0.5, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 70, 80, 100, 125, 150]}
gridcov = {'t': [0, 0.01,0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1]}
def naive_cv(clf, X, y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(clf(0,0,D), param_grid=grid1,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time

def naive_cv_small(clf, X, y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(clf(0,0,D), param_grid=grid1small,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time

def naive_cv_large(clf, X, y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(clf(0,0,D), param_grid=gridlarge,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time

def naive_cv_gtv(X,y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(GTV(0,0,0,D), param_grid=gridGTV,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time
def naive_cv_gtv_small(X,y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(GTV(0,0,0,D), param_grid=gridGTVsmall,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time


def naive_cv_lasso(X, y, D = 0, n_cv = 5):
    gd_sr = GridSearchCV(LA(0,0,D), param_grid=gridlasso,scoring = 'neg_mean_squared_error',cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time

def naive_cv_cov(X, n_cv = 5):
    gd_sr = GridSearchCV(cov_est(0), param_grid=gridcov,scoring = cov_scorer,cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X)
    return result.best_params_, time.time() - start_time



def naive_cv_logit(clf, X, y, D = 0, n_cv = 5):  #Need shuffling 
    gd_sr = GridSearchCV(clf(0,0,D), param_grid=gridlogit,scoring = logit_scorer ,cv=n_cv, n_jobs=-1)
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time

def weighted_incidence(Psi):
    p = Psi.shape[0]
    M = np.zeros((p*(p-1)//2, p))
    G = comp_incidence(p)
    for i in range(p-1):
        for j in range(p):
            if j > i:
                G_row = G[(2*p - 1 - i)*i//2 + j - i - 1]
                G_row[G_row == 1] *= np.sign(Psi[i,j])
                M[(2*p - 1 - i)*i//2 + j - i - 1] = G_row * np.sqrt(abs(Psi[i, j]))
    return np.asarray(M[np.any(M != 0, axis = 1)])

def cor_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    W = np.sqrt(np.diag(np.diagonal(C)))
    C_1 = la.inv(W)@C@la.inv(W)
    return C_1

def cov_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    return C

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





def barbell_signal(m1, m2, start_value, height):
    b = np.zeros(2*m1 + m2)
    b[0:m1] = start_value*np.ones(m1)
    b[m1+m2:2*m1 + m2] = (start_value + height) * np.ones(m1)
    if m2 > 0:
        c = m2+1
        path = [height/2 + height/2 * np.sin(j * np.pi/c) for j in np.arange(-c/2+1, c/2, 1)] 
        path = [x + start_value for x in path]
        b[m1:m1+m2] = np.array(path)
    return b


def D_from_G(G):
    return np.asarray(nx.incidence_matrix(G, oriented = True).T.todense())