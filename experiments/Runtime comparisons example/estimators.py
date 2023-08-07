


import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator

from sklearn.utils.validation import check_is_fitted
from IP_G import*
from CGD_G import*
from ADMM_G import*

def loss_f2(x, y, beta):   
    return cp.norm2(x @ beta - y)**2/2  

def L_pen(beta, l1):
    return l1*cp.norm1(beta)

def SL_pen(beta, l1, l2, D):
    return l1*cp.norm1(beta) + l2*cp.norm2(D @ beta)**2
def EN_pen(beta, l1, l2):
    return l1*cp.norm1(beta) + l2*cp.norm2(beta)**2
def FL_pen(beta, l1, l2, D):
    return l1*cp.norm1(beta) + l2*cp.norm1(D @ beta)

def EE_pen(beta, l1, l2, D):
    return l1*cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2

def GTV_pen(beta, l1, l2, l3, D):
    return l1*cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2 + l3*cp.norm1(beta)

def loss_poi(x, y, beta):
    n = x.shape[0]
    return cp.sum(cp.exp(x @ beta) - cp.multiply(y , x @ beta))   ##/n make l1 l2 large

def loss_logit(x,y, beta):
    n = x.shape[0]
    loglik = cp.sum(cp.multiply(y, x @ beta) - cp.logistic(x @ beta))
    return -loglik        ## not divided by n to make l1 l2 of larger scale

class cov_est(BaseEstimator):
    def __init__(self, t = 0):
        self.t = t
    def fit(self, X):
        S_hat = np.cov(X.T, bias = True)
        S_hat[np.abs(S_hat) < self.t] = 0
        self.S = S_hat
        return self

def cov_scorer(self, X):
    S_hat = np.cov(X.T, bias = True)
    return - np.linalg.norm(self.S - S_hat)


def logit_scorer(self, X, y):
    return -loss_logit(X,y, self.beta).value      #loglik

#def logit_scorer(self, X, y):
    #return self.predict_accuracy(X, y)           #predict accuracy


class estimator(BaseEstimator):
    def __init__(self, l1 = 0, l2 = 0, D = 0):
        self.l1 = l1
        self.l2 = l2
        self.D = D

    def predict(self, X):
        check_is_fitted(self, "beta")
        return X @ self.beta

    def score(self, X, y):
        check_is_fitted(self, "beta")
        return -cp.norm2(y - X @ self.beta).value**2 /X.shape[0]
    
    def l2_risk(self, beta_star):
        check_is_fitted(self, "beta")
            
        return cp.norm2(self.beta - beta_star).value




#######################################################Gaussian###############################################

class OLR(estimator):
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1)/n)) ##ECOS always fail for OLR; might due to scaling
        prob1.solve()
        self.beta = beta1.value
        return self
                           
class LA(estimator):
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + L_pen(beta1, self.l1)))
        prob1.solve()
        self.beta = beta1.value
        return self
class SL(estimator):
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + SL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self
class FL(estimator):
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + FL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self
class EN(estimator):
    def fit(self, X, y):
        n, p = X.shape
    
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + EN_pen(beta1, self.l1, self.l2)))
        prob1.solve()
        self.beta = beta1.value
        return self

class GEN(estimator):
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + EE_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve(solver = cp.ECOS, abstol = 1e-4)
        self.beta = beta1.value
        return self
class GEN_ADMM(estimator):
    def fit(self, X, y):
        self.beta = Gauss_ADMM(X, y, self.D, self.l1, self.l2)
        return self
class GEN_CGD(estimator):
    def fit(self, X, y):
        self.beta = CGD(X,y, self.D, self.l1, self.l2)
        return self

class GEN_IP(estimator):
    def fit(self, X, y):
        self.beta = IP(X,y, self.D, self.l1, self.l2)
        return self

def scorer(est, X, y): # default: greater the better
    return -cp.norm2(y - X @ est.beta).value**2 /X.shape[0]




class GTV(BaseEstimator):
    def __init__(self, l1 = 0, l2 = 0, l3 = 0, D = 0):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.D = D

    def predict(self, X):
        check_is_fitted(self, "beta")
        return X @ self.beta

    def score(self, X, y): # the larger the better
        check_is_fitted(self, "beta")
        return -cp.norm2(y - X @ self.beta).value**2 /X.shape[0]
    
    def l2_risk(self, beta_star):
        check_is_fitted(self, "beta")
            
        return cp.norm2(self.beta - beta_star).value
    def fit(self, X, y):
        n, p = X.shape
        
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + GTV_pen(beta1, self.l1, self.l2, self.l3, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self


