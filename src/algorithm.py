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
    return -loss_logit(X,y, self.beta).value      #loglik


class Estimator(BaseEstimator):
    def __init__(self, l1 = 0, l2 = 0, D = 0):
        self.l1 = l1
        self.l2 = l2
        self.D = D

    def check_D(self):
        if self.D is None:
            print("Incidence matrix is null")

    def predict(self, X):
        check_is_fitted(self, "beta")
        return X @ self.beta

    def score(self, X, y):
        check_is_fitted(self, "beta")
        return -cp.norm2(y - X @ self.beta).value**2 /X.shape[0]

    def l2_risk(self, beta_star):
        check_is_fitted(self, "beta")

        return cp.norm2(self.beta - beta_star).value


class LogRegEstimator(BaseEstimator):
    def __init__(self, l1:float =0, l2:float = 0, D = 0):
        self.l1 = l1
        self.l2 = l2
        self.D = D
    def predict(self, X):
        check_is_fitted(self, "beta")
        return np.round(np.exp(X @ self.beta) / (1 + np.exp(X @ self.beta)))

    def score(self, X, y):
        check_is_fitted(self, "beta")
        return -loss_logit(X, y, self.beta).value

    def predict_accuracy(self, X, y):
        check_is_fitted(self, "beta")
        return 1. - np.sum(np.abs(y - self.predict(X)))/len(y)

class LogRegGenEN(LogRegEstimator):
    def __init__(self, l1=0, l2=0, D=None):
        LogRegEstimator.__init__(self, l1=l1, l2=l2, D=D)

    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(logit_loss(X, y, beta1) +
                           EE_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self

class LogFusedLasso(LogRegEstimator):
    def __init__(self, l1=0, l2=0, D=None):
            LogRegEstimator.__init__(self, l1=l1, l2=l2, D=D)

    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_logit(X, y, beta1) + FL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self

class LogSmoothLasso(LogRegEstimator):
    def __init__(self, l1=0, l2=0, D=None):
                LogRegEstimator.__init__(self, l1=l1, l2=l2, D=D)
    def fit(self, X, y):
        n, p = X.shape
        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_logit(X, y, beta1) +
                           SL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self

class Log_OLR(LogEstimator):
    def fit(self, X, y):
        n, p = X.shape

        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_logit(X,y, beta1)))
        prob1.solve()
        self.beta = beta1.value
        return self

class PoissonEstimator(BaseEstimator):
    def __init__(self, l1 = 0, l2 = 0, D = 0):
        self.l1 = l1
        self.l2 = l2
        self.D = D
    def predict(self, X):
        check_is_fitted(self, "beta")
        return np.exp(X @ self.beta)

    def score(self, X, y):
        check_is_fitted(self, "beta")
        return -loss_poi(X,y, self.beta).value

class Poisson_OUR(PoissonEstimator):
    def fit(self, X, y):
        n, p = X.shape

        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_poi(X, y, beta1) + EE_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self
class Poisson_FL(PoissonEstimator):
    def fit(self, X, y):
        n, p = X.shape

        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_poi(X, y, beta1) + FL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self
class Poisson_SL(PoissonEstimator):
    def fit(self, X, y):
        n, p = X.shape

        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_poi(X, y, beta1) + SL_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self



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
class OUR(estimator):
    def fit(self, X, y):
        n, p = X.shape

        beta1 = cp.Variable(p)
        prob1 = cp.Problem(cp.Minimize(loss_f2(X, y, beta1) + EE_pen(beta1, self.l1, self.l2, self.D)))
        prob1.solve(solver = cp.ECOS, abstol = 1e-4)
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

def scorer(est, X, y): # default: greater the better
    return -cp.norm2(y - X @ est.beta).value**2 /X.shape[0]




class GTV(BaseEstimator):
    def __init__(self, l1:float = 0, l2:float = 0,
                l3 :float= 0, D = None):
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
        prob1 = cp.Problem(cp.Minimize(l2_loss(X, y, beta1) +
                           GTV_penalty(beta1, self.l1, self.l2, self.l3, self.D)))
        prob1.solve()
        self.beta = beta1.value
        return self
