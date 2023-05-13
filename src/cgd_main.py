from lib2to3.pgen2 import grammar
import matplotlib.pyplot as plt
from simulations.examples import SmoothStair, BarbellGraph, GeneralGraph
import numpy as np
import pywt
from numpy import linalg as la
import timeit
from copy import deepcopy
from simulations.covariances import toeplitz_covariance
from simulations.sample_data import gaussian_sample
import networkx as nx
import numpy as np
from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process
from multiprocessing import Process, Value, Array
import timeit
from src.cgd_solver import cgd_solver, primal_dual_preprocessing, cgd_greedy_parallel, cgd_solver_greedy

import time


import sys
import os

os.getcwd()

from src.parallel_fns import print_func, print_func2, test_fn, compute_update, add_one, f, compute_and_update
#print_func("America")

if __name__ == "__main__":  # confirms that the code is under main function
    #stairs = SmoothStair(n_repeat = 150)
    #X, y = gaussian_sample(5000, stairs.n_nodes, beta_star = stairs.beta_star, Psi = stairs.Psi, sigma = 1)
    n = 200
    p = 0.8
    m1 = 30
    m2 = 6
    m = 120
    sizes = [55, 78, 88]
    probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    G = nx.powerlaw_cluster_graph(n, m, p)

    barbell = GeneralGraph(G, 60, 70)

    X, y = gaussian_sample(5000, barbell.n_nodes, beta_star = barbell.beta_star, Psi = np.eye(barbell.n_nodes), sigma = 2)

#test.n_nodes

#plt.plot(test.beta_star)

    #barbell = BarbellGraph(length_chain = 50, size_clique = 50)
    #X, y = gaussian_sample(10000, barbell.n_nodes, beta_star = barbell.beta_star, Psi = np.eye(barbell.n_nodes), sigma = 2)

    dual_params = primal_dual_preprocessing(X, y, barbell.incidence, lambda2 = 1)

    start_time = timeit.default_timer()
    beta = cgd_greedy_parallel(dual_params, lambda1 = 5, eps = 1e-4, max_it = 50000)
    end_time = timeit.default_timer()

    start_time2 = timeit.default_timer()
    beta_normal = cgd_solver(dual_params, lambda1 = 5)
    end_time2 = timeit.default_timer()

    start_time3 = timeit.default_timer()
    beta_greedy = cgd_solver_greedy(dual_params, lambda1 = 5)
    end_time3 = timeit.default_timer()


    #better tracking of process starting and finishing 

    #print("time_elapsed: ", end_time - start_time)
    #print(update_counter.value)

    print('time_elapsed_greedy:', end_time3 - start_time3)
    print('time_elapsed_normal:' , end_time2 - start_time2)
    print('time_elapsed_parallel:', end_time - start_time)
    print("normed diff normal", la.norm(beta_normal - barbell.beta_star)/np.sqrt(len(beta_normal)))
    print("normed diff parallel", la.norm(beta - barbell.beta_star)/np.sqrt(len(beta)))
    print("normed diff greedy", la.norm(beta_greedy[0] - barbell.beta_star)/np.sqrt(len(beta_greedy[0])))



    fig, ax = plt.subplots()
    ax.plot(beta, label = "Greedy")
    ax.plot(beta_normal, label = "Ordinary")
    ax.plot(barbell.beta_star, label = "True")
    plt.legend()
    plt.show()

    
    #print(X)
