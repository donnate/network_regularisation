from lib2to3.pgen2 import grammar
from math import gamma
import matplotlib.pyplot as plt
from simulations.examples import SmoothStair, BarbellGraph, GeneralGraph, Smooth2D
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
from src.cgd_solver import cgd_solver, primal_dual_preprocessing, cgd_greedy_parallel, cgd_solver_greedy, cov_from_G

import os

os.getcwd()

if __name__ == "__main__":  # confirms that the code is under main function
    #X, y = gaussian_sample(5000, stairs.n_nodes, beta_star = stairs.beta_star, Psi = stairs.Psi, sigma = 1)
    '''
    n = 200
    p = 0.8
    m1 = 30
    m2 = 6
    m = 120
    sizes = [55, 78, 88]
    probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    G = nx.powerlaw_cluster_graph(n, m, p)
    '''

    params_list = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #nx.number_of_nodes(smooth2d.G)

    mse_list_normal = []
    mse_list_greedy = []
    mse_list_parallel = []
    runtimes_normal = []
    runtimes_greedy = []
    runtimes_parallel = []
    read_times_parallel = []
    preprocessing_times = []
    
    for params in params_list:
        barbell = Smooth2D(side_len = int(np.sqrt(params)))
        #barbell.n_nodes

        cov_matrix = cov_from_G(barbell.G, 0.1)
        p = nx.number_of_nodes(barbell.G)
        sigma =1 


        g_dagger = np.linalg.pinv(barbell.incidence)
        g_dagger_norms = np.linalg.norm(g_dagger, axis = 0)
        rho_gamma = np.max(g_dagger_norms)
        
        gamma_max_cov = np.max(np.linalg.eigh(cov_matrix)[0])


        lambda1_opt = 32*sigma*rho_gamma*np.sqrt((gamma_max_cov * np.log(p))/nx.number_of_nodes(barbell.G))

        lambda2_opt = lambda1_opt/(16 * np.max(barbell.incidence@barbell.beta_star.flatten()))
        

        X, y = gaussian_sample(p, nx.number_of_nodes(barbell.G), beta_star = barbell.beta_star.flatten(), Psi = cov_matrix, sigma = sigma)

        start_time4 = timeit.default_timer()
        dual_params = primal_dual_preprocessing(X, y, barbell.incidence, lambda2 = lambda2_opt)
        end_time4 = timeit.default_timer()
        preprocessing_times.append(end_time4 - start_time4)


        start_time2 = timeit.default_timer()
        beta_normal = cgd_solver(dual_params, lambda1 = lambda1_opt)
        end_time2 = timeit.default_timer()
        mse_list_normal.append(la.norm(beta_normal - barbell.beta_star.flatten())/np.sqrt(len(beta_normal)))
        runtimes_normal.append(end_time2 - start_time2)

        start_time3 = timeit.default_timer()
        beta_greedy = cgd_solver_greedy(dual_params, lambda1 = lambda1_opt)
        end_time3 = timeit.default_timer()
        mse_list_greedy.append(la.norm(beta_greedy[0] - barbell.beta_star.flatten())/np.sqrt(len(beta_greedy[0])))
        runtimes_greedy.append(end_time3 - start_time3)

        start_time = timeit.default_timer()
        beta, read_time = cgd_greedy_parallel(dual_params, lambda1 = lambda1_opt, eps = 1e-5)
        end_time = timeit.default_timer()
        mse_list_parallel.append(la.norm(beta - barbell.beta_star.flatten())/np.sqrt(len(beta)))
        read_times_parallel.append(read_time)
        runtimes_parallel.append(end_time - start_time)

        print(f"finished computation for params {params}")
        print(f"mse_list_normal{mse_list_normal}")
        print(f"mse_list_greedy{mse_list_greedy}")
        print(f"mse_list_parallel{mse_list_parallel}")
        print(f"runtimes_normal{runtimes_normal}")
        print(f"runtimes_greedy{runtimes_greedy}")
        print(f"runtimes_parallel{runtimes_parallel}")
        print(f"read_times_parallel{read_times_parallel}")
        print(f"preprocessing_times{preprocessing_times}")

        #better tracking of process starting and finishing 

        #print("time_elapsed: ", end_time - start_time)
        #print(update_counter.value)

        #print('time_elapsed_greedy:', end_time3 - start_time3)
        #print('time_elapsed_normal:' , end_time2 - start_time2)
        #print('time_elapsed_parallel:', end_time - start_time)
        #print("normed diff normal", la.norm(beta_normal - barbell.beta_star)/np.sqrt(len(beta_normal)))
        #print("normed diff parallel", la.norm(beta - barbell.beta_star)/np.sqrt(len(beta)))
        #print("normed diff greedy", la.norm(beta_greedy[0] - barbell.beta_star)/np.sqrt(len(beta_greedy[0])))

        #fig, ax = plt.subplots()
        #ax.plot(beta, label = "Greedy")
        #ax.plot(beta_normal, label = "Ordinary")
        #ax.plot(barbell.beta_star, label = "True")
        #plt.legend()
        #plt.show()