#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import random
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def avg(mat, N=2):
    """
    calculate sample average for every N steps. 

    reference:
    https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
    """
    cum = np.cumsum(mat,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = mat.shape[0] % N
    if remainder != 0:
        if remainder < mat.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

def proj(mat, coord, proj_coord, k=10):
    """
    project data defined by mat from coordinate system 1 to coordinate system 2.

    Args:
    - mat:        2D data matrix         [ n_days, n_from_locations ]
    - coord:      from coordinate system [ n_from_locations, 2 ]
    - proj_coord: to coordinate system   [ n_to_locations, 2 ]
    - k:          find the nearest k points
    """
    dist      = euclidean_distances(proj_coord, coord) # [ n_to_locations, n_from_locations ]
    argdist   = np.argsort(dist, axis=1)               # [ n_to_locations, n_from_locations ]
    neighbors = argdist[:, :k]                         # [ n_to_locations, k ]
    # projection
    N, K      = mat.shape
    proj_K    = proj_coord.shape[0]
    proj_mat  = np.zeros((N, proj_K)) 
    for t in range(N):
        for loc in range(proj_K):
            proj_mat[t, loc] = mat[t, neighbors[loc]].mean()

    return proj_mat