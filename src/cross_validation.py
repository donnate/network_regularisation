import numpy as np
import time
import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from src.estimators import GTVEstimator, LassoEstimator

GRID_LASSO = {'l1': [0,0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5,
                    7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100], 'l2':[0]}
GRID1 = {'l1': [0,0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5,
               10, 12.5, 15, 20],
         'l2': [0,0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5,
                10, 12.5, 15, 20]}
GRID1_SMALL = {'l1': [0,0.1, 1, 10, 20, 50, 100, 200],
               'l2': [0,0.1, 1, 10, 20, 50, 100, 200]}
GRID_GTV = {'l1': [0, 0.001, 0.01,0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20], # for 2D
            'l2': [0, 0.001, 0.01,0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15, 20],
            'l3': [0]} #save time
GRID_GTV_SMALL = {'l1': [0, 0.1, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 25],
                  'l2': [0, 0.1, 0.5, 1, 2, 5, 3,  7.5, 10, 15, 25],
                  'l3': [0, 0.1, 0.25, 0.5, 1, 2, 5]}  #1D
GRID_LOGIT  = {'l1': [0, 0.005, 0.01, 0.1, 0.2, 0.5],
               'l2': [0, 0.005, 0.01, 0.1, 0.2, 0.5]}
GRID_LARGE = {'l1': [0 ,0.1, 0.5, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25,
                     30, 40, 50, 70, 80, 100, 125, 150],
              'l2': [0,0.1, 0.5, 1, 2, 5, 7.5, 10, 12.5, 15, 20,
                     25, 30, 40, 50, 70, 80, 100, 125, 150]}
GRID_COV = {'t': [0, 0.01, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3,
                  0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1]}


def naive_cv(clf, X, y, D = 0, n_cv = 5, grid=GRID1_SMALL, solver=None,
             family='normal'):
    if family == 'normal':
        gd_sr = GridSearchCV(clf(D=D, family='normal', solver=solver),
                             param_grid=grid, scoring = 'neg_mean_squared_error',
                             cv=n_cv, n_jobs=-1)
    elif family == 'poisson':
        gd_sr = GridSearchCV(clf(D=D, family='poisson'),
                             param_grid=grid, scoring = 'neg_mean_poisson_deviance',
                             cv=n_cv, n_jobs=-1)
    elif family == 'binomial':
        gd_sr = GridSearchCV(clf(D=D, family='binomial'),
                                 param_grid=grid, scoring = 'f1',
                                 cv=n_cv, n_jobs=-1)
    else:
        raise ValueError('Exponential family not implemented yet')
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time


def naive_cv(clf, X, y, D = 0, n_cv = 5, grid=GRID1_SMALL, solver=None,
             family='normal', shuffle = True):
    kf = KFold(n_splits=n_cv, shuffle=shuffle, random_state=None)


    if family == 'normal':
        gd_sr = GridSearchCV(clf(D=D, family='normal', solver=solver),
                             param_grid=grid, scoring = 'neg_mean_squared_error',
                             cv=kf, n_jobs=-1)
    elif family == 'poisson':
        gd_sr = GridSearchCV(clf(D=D, family='poisson'),
                             param_grid=grid, scoring = 'neg_mean_poisson_deviance',
                             cv=kf, n_jobs=-1)
    elif family == 'binomial':
        gd_sr = GridSearchCV(clf(D=D, family='binomial'),
                                 param_grid=grid, scoring = 'f1',
                                 cv=kf, n_jobs=-1)
    else:
        raise ValueError('Exponential family not implemented yet')
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time



def naive_cv_gtv(X,y, D = 0, n_cv = 5, grid=GRID_GTV, family='normal'):
    if family == 'normal':
        gd_sr = GridSearchCV(GTVEstimator(0, 0, 0, D, family='normal'),
                             param_grid=grid, scoring = 'neg_mean_squared_error',
                             cv=n_cv, n_jobs=-1)
    elif family == 'poisson':
        gd_sr = GridSearchCV(GTVEstimator(0, 0, 0, D, family='poisson'),
                             param_grid=grid, scoring = 'neg_mean_poisson_deviance',
                             cv=n_cv, n_jobs=-1)
    elif family == 'binomial':
        gd_sr = GridSearchCV(GTVEstimator(0, 0, 0, D, family='binomial'),
                                 param_grid=grid, scoring = 'f1',
                                 cv=n_cv, n_jobs=-1)
    else:
        raise ValueError('Exponential family not implemented yet')
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time


def naive_cv_lasso(X, y, D = 0, n_cv = 5, grid=GRID_LASSO, family='normal'):
    if family == 'normal':
        gd_sr = GridSearchCV(LassoEstimator(0, 0, D, family='normal'),
                             param_grid=grid, scoring = 'neg_mean_squared_error',
                             cv=n_cv, n_jobs=-1)
    elif family == 'poisson':
        gd_sr = GridSearchCV(LassoEstimator(0, 0, D, family='poisson'),
                             param_grid=grid, scoring = 'neg_mean_poisson_deviance',
                             cv=n_cv, n_jobs=-1)
    elif family == 'binomial':
        gd_sr = GridSearchCV(LassoEstimator(0, 0, D, family='binomial'),
                                 param_grid=grid, scoring = 'f1',
                                 cv=n_cv, n_jobs=-1)
    else:
        raise ValueError('Exponential family not implemented yet')
    start_time = time.time()
    result = gd_sr.fit(X,y)
    return result.best_params_, time.time() - start_time
