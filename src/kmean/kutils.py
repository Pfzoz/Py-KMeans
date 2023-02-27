import numpy as np

class InvalidShape(Exception):
    pass

def normalize(x : np.ndarray) -> np.ndarray:
    if x.ndim >= 2:
        for i in range(x.shape[0]):
            x[i] = x[i]/max(x[i])
    else:
        x = x/max(x)
    return x

def euclid(point_a, point_b) -> float:
    if len(point_a) != len(point_b):
        print("Shape differs for euclidean distance calc.")
        raise InvalidShape
    sum = 0
    for x, y in zip(point_a, point_b):
        sum += (x-y)**2
    return np.sqrt(sum)
