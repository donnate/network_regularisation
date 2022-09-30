import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pywt
import random
import time


class Example:
    def __init__(self):
        self.G = nx.Graph()
        self.n_nodes = nx.number_of_nodes(self.G)
        self.coordinates = None

    def plot_signal(self):
        plt.figure()
        plt.plot(range(len(self.y)), self.y)
        plt.show()

    def draw_graph(self):
        if self.coordinates is None:
            plt.figure()
            nx.draw(self.G)
            plt.show()
        else:
            plt.figure()
            nx.draw(self.G, pos = self.coordinates, node_size=3, node_color='r')
            plt.show()

    def plot_adjacency(self):
        return

class SmoothStair(Example):
    def __init__(self, slope_length: int=3, slope_height: float=3.,
                 step_length: int=4, start_value: float=0.,
                 n_repeat: int=10):

        pattern = [slope_height/2  * (1 + np.sin(j * np.pi/slope_length))
                   for j in np.arange(-slope_length/2, slope_length/2, 1)] + step_length * [slope_height]
        temp_y = []
        for i in range(n_repeat):
            temp_y += [x + slope_height*i for x in pattern]
        self.y =  np.array([x + start_value for x in temp_y][slope_length: ])
        self.G = nx.grid_graph(dim=((len(self.y), 1)))
        self.n_nodes = nx.number_of_nodes(self.G)
        self.coordinates = {n : (k, 0) for k, n in enumerate(self.G.nodes)}
        #self.incidence = np.asarray(incidence_matrix(self.G).T.todense())

class Toeplitz(Example):
    def __init__(self, a:float=3, p:int=4):
        self.y = np.exp(np.log(a) * np.abs(np.subtract.outer(range(p),
                        range(p))))


class BarbellGraph(Example):
    def __init__(self, m1, m2, start_value, height):
            b = np.zeros(2 * m1 + m2)
            b[0: m1] = start_value * np.ones(m1)
            b[(m1 + m2):(2 * m1 + m2)] = (start_value + height) * np.ones(m1)
            if m2 > 0:
                c = m2+1
                path = [height/2 + height/2 * np.sin(j * np.pi/c)
                        for j in np.arange(-c/2 + 1, c/2, 1)]
                path = [x + start_value for x in path]
                b[m1 :(m1 + m2)] = np.array(path)

# def chain_incidence(p):
#     return scipy.sparse.diags([np.ones(p), -np.ones(p-1)], [0, 1]).toarray()[0:p-1,:]


# def gauss_sample(n, p, beta_star, Psi, sigma, set_seed = 1):
#     random.seed(set_seed)
#     X = np.random.multivariate_normal(mean = np.zeros(p), cov=Psi, size=n)
#     y = X.dot(beta_star) + sigma * np.random.normal(size=n)
#     return X, y

class Weighted(Example):
    def __init__(self, Psi):
        p = Psi.shape[0]
        M = np.zeros((p * (p - 1) // 2, p))
        G = comp_incidence(p)
        for i in range(p-1):
            for j in range(p):
                if j > i:
                    G_row = G[(2*p - 1 - i)*i // 2 + j - i - 1]
                    G_row[G_row == 1] *= np.sign(Psi[i,j])
                    M[(2 * p - 1 - i)*i//2 + j - i - 1] = G_row * np.sqrt(abs(Psi[i, j]))
        self.y =  np.asarray(M[np.any(M != 0, axis = 1)])

## Caution: may have bugs
class Smooth2D(Example):
    def __init__(self, side_len, disk_radi, period, height, start_value):
        if side_len % 2 == 0:
            left_end = -(side_len//2) + 1
        else:
            left_end = -(side_len//2)
        x, y = np.meshgrid(np.linspace(left_end, side_len//2, side_len),
                           np.linspace(left_end, side_len//2, side_len))
        ds = np.sqrt(x**2 + y**2)
        ds = pywt.threshold(ds, disk_radi, 'soft')
        ds[abs(ds) > period] = -period
        ds = np.array([x + period/2 for x in ds])
        part = np.sin(np.pi*ds/period)
        self.y = np.array([height/2 * x + start_value + height/2 for x in part])


class CompleteGraph(Example):
    def __init__(self, side_len, disk_radi, period, height, start_value):
        self.y = 0

class KGrid(Example):
    def __init__(self, side_len, disk_radi, period, height, start_value):
        self.y = 0
