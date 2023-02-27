import numpy as np
from numpy.random import random
from matplotlib import pyplot as plt
from kmean.kmeans import KMeans
from kmean.kutils import normalize

simple_data = np.array(
    [
        [2, 3, 2, 9, 10, 4, 11, 12, 10.5, 1],
        [0.5, 1.5, 2, 3, 2, 10, 8, 4, 2, 10]
    ]
)

simple_data = normalize(simple_data)
print(simple_data, '\n\n')

myModel = KMeans()

myModel.fit(simple_data, k=5, epochs=1000)
myModel.plot_2D_show()