import numpy as np
from matplotlib import pyplot as plt
from numpy.random import choice, random
from .kutils import *

class Cluster:

    def __init__(self, core : list[float], family : list[list[float]] = None):
        self.core = np.array(core)
        if family is None:
            self.family = []
        else:
            self.family = family
    
    def __repr__(self) -> str:
        return f"{self.core}: {self.family}"

    def get_mean(self) -> np.ndarray:
        all_points = self.family + [np.array(self.core)]
        mean_point = sum(all_points)/len(all_points)
        return mean_point

class KMeans:

    def __init__(self, clusters : list[Cluster] = None):
        if clusters is None:
            self.clusters = None
        else:
            self.clusters = np.array(clusters)
    
    def fit(self, x : np.ndarray, k : int, epochs : int = 100):
        original = x
        if x.ndim == 2:
            if self.clusters is None:
                clusters = []
                for i in range(k):
                    splicer = choice(x.shape[1])
                    cluster = Cluster(core=x[:, splicer])
                    clusters.append(cluster)
                    x = np.concatenate((x[:, :splicer], x[:, splicer+1:]), axis=1)
                self.clusters = np.array(clusters)
            for epoch in range(epochs):
                print(f"Epoch {epoch}")
                for i in range(x.shape[1]):
                    point_a = x[:, i]
                    distance = euclid(point_a, self.clusters[0].core)
                    selected_cluster = self.clusters[0]
                    for cluster in self.clusters[1:]:
                        new_distance = euclid(point_a, cluster.core)
                        if new_distance < distance:
                            distance = new_distance
                            selected_cluster = cluster
                    selected_cluster.family.append(point_a)
                if epoch == epochs-1:
                    break
                for cluster in self.clusters:
                    flipped = cluster.core.reshape((x.shape[0], 1))
                    x = np.concatenate((x, flipped), axis=1)
                    cluster.core = cluster.get_mean()
                    cluster.family.clear()
                
    
    def plot_2D_show(self):
        x1 = []
        x2 = []
        colors = []
        sizes = []
        for cluster in self.clusters:
            chosen_color = (random(), random(), random())
            for point in cluster.family:
                if point.shape[0] != 2:
                    print("Different shape required for 2D plot.")
                    exit()
                x1.append(point[0])
                x2.append(point[1])
                colors.append(chosen_color)
                sizes.append(50)
            x1.append(cluster.core[0])
            x2.append(cluster.core[1])
            colors.append(chosen_color)
            sizes.append(250)
        plt.scatter(x1, x2, s=sizes, c=colors)
        plt.show()
