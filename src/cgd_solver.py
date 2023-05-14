#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pywt
from numpy import linalg as la
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process
from multiprocessing import Process, Value, Array
from src.parallel_fns import print_func, print_func2, test_fn, compute_update, add_one, f, compute_and_update
import time
import timeit
import networkx as nx

def primal_dual_preprocessing(X, y, Gamma, lambda2):
    m, p = Gamma.shape
    X_til, y_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma)), np.concatenate((y, np.zeros(m)))
    X_til_pinv = la.pinv(X_til)

    y_v = X_til @ X_til_pinv @ y_til
    Gamma_v = Gamma @ X_til_pinv
    
    Q = Gamma_v @ Gamma_v.T
    b = Gamma_v @ y_v

    return (m, X_til_pinv, Q, b, y_v, Gamma_v)

def cov_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    return C


def cgd_solver(preprocessed_params, lambda1, eps = 1e-5, max_it = 5000000):
    
    m, X_til_pinv, Q, b, y_v, Gamma_v = preprocessed_params
    u = np.zeros(m)
    n_iter = 0
    prev_u = 0 # For stopping criteria
    while True:
        n_iter += 1
        if n_iter >= max_it:
            #raise ValueError("Iterations exceed max_it")
            print("Iterations exceed max_it")
            return X_til_pinv @ (y_v - Gamma_v.T @ u)
        for i in range(m):
            if Q[i, i] > 1e-4:
                t = 1/Q[i,i] * (b[i] - np.dot(np.delete(Q[i], i), np.delete(u, i)))
            else:
                t = 0

            u[i] = np.sign(t) * min(np.abs(t), lambda1)   #there should be better truncation methods

        if la.norm(u - prev_u) <= eps:
            break

        prev_u = np.copy(u)   # Recall array is similar to list

    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta


def project_op(vector, param): 
    vector[vector > param] = param
    vector[vector < -param] = -param
    return vector

def cgd_solver_greedy(preprocessed_params, lambda1, eps = 1e-5, max_it = 5000000):

    m, X_til_pinv, Q, b, y_v, Gamma_v = preprocessed_params

    u = np.zeros(m)
    n_iter = 0
    comps = 0
    #print(m)
    prev_u = np.zeros(m) # For stopping criteria
    gradient = -b
    while True:
        n_iter += 1
        if n_iter >= max_it:
            #raise ValueError("Iterations exceed max_it")
            print("Iterations exceed max_it")
            return X_til_pinv @ (y_v - Gamma_v.T @ u), n_iter, comps
        projected_gradient = u - project_op(u - gradient, lambda1)
        #projected_gradient = project_op(u - gradient, lambda1)
        greedy_coord = np.argmax(np.abs(projected_gradient))
        i = greedy_coord
        delta = min(max(u[i] - ((1/Q[i,i]) * gradient[i]), -lambda1), lambda1) - u[i]
        gradient += delta *Q[i]
        u[i] += delta
        
        #add back well conditioned block...? 
        #print( "i is " + str(i))
        #print(delta)

        #u[i] = np.sign(t) * min(np.abs(t), lambda1)   #there should be better truncation methods
        if (la.norm(u - prev_u) <= eps) & (n_iter>10):
            print("break reason 1")
            break
        
        #print(gradient)
        if la.norm(gradient) <= eps:
            print("break reason gradient")
            break

        prev_u = np.copy(u)   # Recall array is similar to list
        

    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta, n_iter, comps


def cgd_greedy_parallel(preprocessed_params, lambda1, eps = 1e-5, max_it = 5000000): 
    m, X_til_pinv, Q, b, y_v, Gamma_v = preprocessed_params

    n_iter = 0
    update_loops = 0
    prev_u = 0 # For stopping criteria
    max_it = 50000
    processors = 5 #nprocs #be careful of how many processors to use
    update_counter = Value('i', 0)
    u_arr = Array('f', np.zeros(m))
    grad_arr = Array('f', np.copy(-b))
    start_time = timeit.default_timer()
    Q_arr = Array('f', np.copy(Q.reshape(1, Q.shape[0]*Q.shape[1]).flatten()))
    read_time = timeit.default_timer() - start_time
    #print("read_time:", end_time - start_time)



    procs = []

    split, mod = divmod(m, processors)

    for i in range(processors): 
        p = Process(target=compute_and_update, args=(u_arr, grad_arr, Q_arr, update_counter, eps, lambda1, i*split+min(i, mod), (i+1)*split+min(i+1, mod), max_it))
        procs.append(p)
        p.start()
        print(f"new process index {i} starting now")

    for proc in procs:
        proc.join()

    beta = X_til_pinv @ (y_v - Gamma_v.T @ np.array(u_arr[:]))
    return beta, read_time



# ### Generate stair shape toy example

# In[6]:
