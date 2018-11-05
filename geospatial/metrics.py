from haversine import haversine
import numpy as np


def haversine_distance(x, y):
    vector_alpha = (x[0], x[1])
    vector_beta = (y[0], y[1])
    point_dist = haversine(vector_alpha, vector_beta)*0.539956803

    try:
        feature_dist = np.linalg.norm(x[2:] - y[2:])
    except IndexError:
        feature_dist = 0
    return point_dist + feature_dist