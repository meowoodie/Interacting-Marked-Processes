#!/usr/bin/env python
# -*- coding: utf-8 -*-

import arrow
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

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

# def scale_down_data(data, coord, k=20):
#     """
#     cluster coords into k components.

#     Args:
#     - data:     data matrix           [ N, n_locations ]
#     - coord:    coordinate system     [ n_locations, 2 ]
#     Return:
#     - newdata:  scaled data matrix    [ N, k ]
#     - newcoord: new coordinate system [ k, 2]
#     """
#     N, K    = data.shape
#     kmeans  = KMeans(n_clusters=k, random_state=0).fit(coord)
#     newdata = np.zeros((N, k))
#     for _id in range(k):
#         loc_inds        = np.where(kmeans.labels_ == _id)[0]
#         newdata[:, _id] = data[:, loc_inds].mean(axis=1)
#     return newdata, kmeans.cluster_centers_

# def plot_gradient_line(handler, start_point, end_point, value, cmap):
#     cmap = 
#     handler.plot([start_point[1], end_point[1]], [start_point[0], end_point[j, 0]], latlon=True, c=edge, linewidth=1.)

def colorline(ax, 
    start_point, end_point, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), steps=10,
    linewidth=1., alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    path  = mpath.Path(np.stack([start_point, end_point]))
    verts = path.interpolated(steps=steps).vertices
    x, y  = verts[:, 0], verts[:, 1]
    z     = np.linspace(0, 1, len(x))

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, 
                              linewidth=linewidth, alpha=alpha)

    # ax = plt.gca()
    ax.add_collection(lc)

    return lc

# https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap