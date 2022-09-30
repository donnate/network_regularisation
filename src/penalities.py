import numpy as np
import cvxpy as cp

def lasso_penalty(beta, l1):
    return l1 * cp.norm1(beta)

def smoothlasso_penalty(beta, l1, l2, D):
    return l1 * cp.norm1(beta) + l2 * cp.norm2(D @ beta)**2

def elasticnet_penalty(beta, l1, l2):
    return l1 * cp.norm1(beta) + l2 * cp.norm2(beta)**2

def fusedlasso_penalty(beta, l1, l2, D):
    return l1 * cp.norm1(beta) + l2 * cp.norm1(D @ beta)

def ee_penalty(beta, l1, l2, D):
    return l1*cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2

def gtv_penalty(beta, l1, l2, l3, D):
    return l1*cp.norm1(D @ beta) + l2*cp.norm2(D @ beta)**2 + l3 * cp.norm1(beta)
