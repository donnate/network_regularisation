import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pywt
import random
import time


from sklearn.cluster import SpectralClustering

from simulations.covariances import toeplitz_covariance


class Example:
    def __init__(self):
        self.G = nx.Graph()
        self.n_nodes = nx.number_of_nodes(self.G)
        self.coordinates = None

    def plot_signal(self):
        plt.figure()
        plt.plot(range(len(self.beta_star)), self.beta_star)
        plt.show()

    def draw_graph(self, node_size=1,
                   axis=False):
        fig, ax = plt.subplots()
        if self.coordinates is None:

            nx.draw(self.G, node_color=self.beta_star, cmap='Spectral', ax=ax)
        else:
            plt.figure()
            nx.draw(self.G, pos = self.coordinates, node_color=self.beta_star,
                    node_size = node_size,
                    cmap='Spectral', ax =ax)
        if axis:
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def plot_adjacency(self):
        return

class SmoothStair(Example):
    def __init__(self, slope_length: int=3, slope_height: float=3.,
                 step_length: int=4, start_value: float=0.,
                 n_repeat: int=10, a:float=3, p:int=4
                 ):

        pattern = [slope_height/2  * (1 + np.sin(j * np.pi/slope_length))
                   for j in np.arange(-slope_length/2, slope_length/2, 1)] + step_length * [slope_height]
        temp_y = []
        for i in range(n_repeat):
            temp_y += [x + slope_height*i for x in pattern]
        self.beta_star =  np.array([x + start_value for x in temp_y][slope_length: ])
        self.G = nx.grid_graph(dim=((len(self.beta_star), 1)))
        self.n_nodes = nx.number_of_nodes(self.G)
        self.coordinates = {n : (k, self.beta_star[k]) for k, n in enumerate(self.G.nodes)}
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())
        self.Psi = toeplitz_covariance(a, self.n_nodes)

class GeneralGraph(Example):
    def __init__(self,G, length_chain, size_clique, start_value: float=0., height: float=1.,
                nb_clusters=2, sigma_gen=0.3, signal_type ="tree"):
        self.G = G
        self.coordinates = None
        self.n_nodes = nx.number_of_nodes(self.G)
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())
        self.beta_star = np.zeros(self.n_nodes)
        #### Define what kind of signal we want
        if signal_type == "piecewise_cnst":
            ### Cluster the nodes
            ### decrease sigma_gen for more smoothness
            clustering = SpectralClustering(n_clusters=nb_clusters, assign_labels='discretize',random_state=0,
                                           affinity='precomputed').fit(nx.adjacency_matrix(G))
            for i in np.unique(clustering):
                self.beta_star[np.where(clustering.labels_ == i)] = np.random.normal(loc=np.random.normal(loc=0.0, scale=sigma_gen))
        else:
            ### Find MST
            T = nx.minimum_spanning_tree(G)
            edges_T = nx.incidence_matrix(T)
            #### generate edges differences:
            edges_diff = np.random.normal(loc=0, scale=sigma_gen, size=edges_T.shape[1])
            self.beta_star = np.linalg.pinv(edges_T.todense()).T.dot(edges_diff)
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())

class BarbellGraph(Example):
    def __init__(self, length_chain, size_clique, start_value: float=0., height: float=1.):
        self.G = nx.barbell_graph(size_clique, length_chain)
        self.coordinates = None
        self.n_nodes = nx.number_of_nodes(self.G)
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())
        self.beta_star = start_value * np.ones(self.n_nodes)
        self.beta_star[(size_clique + length_chain):(2 * size_clique + length_chain)] = (start_value + height)
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())
        if length_chain > 0:
            c = length_chain + 1
            path = [height/2 + height/2 * np.sin(j * np.pi/c)
                    for j in np.arange(-c/2 + 1, c/2, 1)]
            path = [x + start_value for x in path]
            self.beta_star[size_clique :(size_clique + length_chain)] = path

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
        self.beta_star =  np.asarray(M[np.any(M != 0, axis = 1)])


class Smooth2D(Example):
    def __init__(self, side_len: int=30, disk_radi: float =3.,
                 period: float=3., height: float=4., start_value: float=2):
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
        self.beta_star = np.array([height/2 * x + start_value + height/2 for x in part])
        self.G = nx.grid_graph(dim=((len(x), len(y))))
        self.incidence = np.asarray(nx.incidence_matrix(self.G, oriented=True).T.todense())
        self.coordinates = {n : n for k, n in enumerate(self.G.nodes)}

    def plot_3d():
        return
