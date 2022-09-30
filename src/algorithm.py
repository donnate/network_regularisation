import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from losses import *
from penalties import *
from algorithm.IP_G import*
from algorithm.CGD_G import*
from algorithm.ADMM_G import*


class CovEst(BaseEstimator):
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
    return -logit_loss(X,y, self.beta).value      #loglik


class Estimator(BaseEstimator):
    def __init__(self, l1 = 0, l2 = 0, D = 0,
                 familiy:str = 'normal'):
        self.l1 = l1
        self.l2 = l2
        self.D = D

    def check_D(self):
        if self.D is None:
            print("Incidence matrix is null")

    def predict(self, X):
        check_is_fitted(self, "beta")
        if self.family == 'normal':
            return X @ self.beta
        elif self.family == 'poisson':
            return np.exp(X @ self.beta)
        elif self.family == 'binomial':
            return np.round(np.exp(X @ self.beta)/(1 +np.exp(X @ self.beta)))
        else:
            print("Not implemented yet")
            return

    def score(self, X, y):
        check_is_fitted(self, "beta")
        if self.family == 'normal':
            return -cp.norm2(y - X @ self.beta).value**2 /X.shape[0]
        elif self.family == 'binomial':
            return -logit_loss(X, y, self.beta).value
        elif self.family == 'poisson':
            return -poisson_loss(X,y, self.beta).value

    def l2_risk(self, beta_star):
        check_is_fitted(self, "beta")
        return cp.norm2(self.beta - beta_star).value

    def predict_accuracy(self, X, y):
        check_is_fitted(self, "beta")
        if self.family == 'binomial':
            return 1. - np.sum(np.abs(y - self.predict(X)))/len(y)
        else:
            return None


class NaiveEstimator(Estimator):
    def __init__(self, l1=0, l2=0, D=None, family='normal'):
        Estimator.__init__(self, l1=l1, l2=l2, D=D,
                                 family=family)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                               EE_pen(beta1, self.l1, self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self


class LassoEstimator(Estimator):
    def __init__(self, l1=0, family='binomial'):
        Estimator.__init__(self, l1=l1, l2=None, D=None,
                           family=family)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           lasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           lasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           lasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self

class FusedLassoEstimator(Estimator):
    def __init__(self, l1=0, l2=0, D=None, family='normal'):
            Estimator.__init__(self, l1=l1, l2=l2, D=D,
                                     family=family)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           fusedlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           fusedlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           fusedlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self


class SmoothLassoEstimator(Estimator):
    def __init__(self, l1=0, l2=0, D=None, family='binomial'):
                Estimator.__init__(self, l1=l1, l2=l2, D=D,
                                         family=family)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           smoothlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           smoothlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           smoothlasso_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self



class GenElasticNetEstimator(Estimator):
    def __init__(self, l1=0, l2=0, D=None, family='binomial'):
                Estimator.__init__(self, l1=l1, l2=l2, D=D,
                                         family=family)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           ee_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           ee_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           ee_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self



class ElasticNetEstimator(Estimator):
    def __init__(self, l1=0, l2=0, D=None, ):
                Estimator.__init__(self, l1=l1, l2=l2, D=None,
                                         family='binomial')
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           elasticnet_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           elasticnet_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           elasticnet_penalty(beta1, self.l1,
                                           self.l2, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self


class OUR_ADMM(estimator):
    def fit(self, X, y):
        self.beta = Gauss_ADMM(X, y, self.D, self.l1, self.l2)
        return self
class OUR_CGD(estimator):
    def fit(self, X, y):
        self.beta = CGD(X,y, self.D, self.l1, self.l2)
        return self

class OUR_IP(estimator):
    def fit(self, X, y):
        self.beta = IP(X,y, self.D, self.l1, self.l2)
        return self

class GTV(BaseEstimator):
    def __init__(self, l1:float = 0, l2:float = 0,
                l3 :float= 0, D = None, family:str = "normal"):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.D = D
        self.family = family

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
        if self.family == 'binomial':
            prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                                           gtv_penalty(beta1, self.l1,
                                           self.l2, self.l3,
                                        self.D)))
        elif self.family == 'poisson':
            prob1 = cp.Problem(cp.Minimize(poisson_loss(X, y, beta1) +
                                           gtv_penalty(beta1, self.l1, self.l2,
                                           self.l3, self.D)))
        elif self.family == 'normal':
            prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) / n) +
                                           gtv_penalty(beta1, self.l1, self.l2,
                                           self.l3, self.D)))
        else:
            raise ValueError('Exponential family not implemented yet')
        prob1.solve()
        self.beta = beta1.value
        return self