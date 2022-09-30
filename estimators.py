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
