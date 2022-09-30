import numpy as np
import cvxpy as cp


def l2_loss(x, y, beta):
    return cp.norm2(x @ beta - y)**2 / 2

def poisson_loss(x, y, beta):
    n = x.shape[0]
    return cp.sum(cp.exp(x @ beta) - cp.multiply(y , x @ beta))   ##/n make l1 l2 large

def logit_loss(x, y, beta):
    n = x.shape[0]
    loglik = cp.sum(cp.multiply(y, x @ beta) - cp.logistic(x @ beta))
    return -loglik/n
