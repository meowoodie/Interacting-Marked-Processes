#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages



def plot_data_on_map(data, locs, filename, dmin=None, dmax=None):
    """
    Args:
    - data: [ n_locs ]

    References:
    - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
    - https://matplotlib.org/basemap/users/geography.html
    - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
    """

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # Make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    mins, maxs = 5, 300
    dmin, dmax = data.min() if dmin is None else dmin, data.max() if dmax is None else dmax
    print(dmin, dmax)
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c="r")
    # sct  = m.scatter(locs[_id, 1], locs[_id, 0], latlon=True, alpha=1., s=10, c="b")
    # legend
    handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
        func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
    plt.title(filename)
    plt.legend(handles, labels, loc="lower left", title="Num of outages")

    plt.savefig("imgs/%s.pdf" % filename)

def plot_data_on_linechart(data, filename, dmin=None, dmax=None):
    """
    Args:
    - data: [ n_timeslots ]
    """

    dates = [
        "2018-03-01", "2018-03-02", "2018-03-03", "2018-03-04", "2018-03-05",
        "2018-03-06", "2018-03-07", "2018-03-08", "2018-03-09", "2018-03-10",
        "2018-03-11", "2018-03-12", "2018-03-13", "2018-03-14", "2018-03-15" ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("imgs/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.arange(len(data)), data, c="#cb416b", linewidth=3, linestyle="-", label="Real", alpha=.8)
        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(0, len(data), 24 / 3), dates, rotation=90)
        plt.xlabel(r"Date")
        plt.ylabel(r"Number of outages")
        # plt.legend(loc='upper left', fontsize=13)
        plt.title(filename)
        fig.tight_layout()
        pdf.savefig(fig)

def plot_2data_on_linechart(data1, data2, filename, dmin=None, dmax=None):
    """
    Args:
    - data: [ n_timeslots ]
    """

    dates = [
        "2018-03-01", "2018-03-02", "2018-03-03", "2018-03-04", "2018-03-05",
        "2018-03-06", "2018-03-07", "2018-03-08", "2018-03-09", "2018-03-10",
        "2018-03-11", "2018-03-12", "2018-03-13", "2018-03-14", "2018-03-15" ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("imgs/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.arange(len(data1)), data1, c="#cea2fd", linewidth=3, linestyle="--", label="Real", alpha=.8)
        ax.plot(np.arange(len(data2)), data2, c="#677a04", linewidth=3, linestyle="--", label="Prediction", alpha=.8)
        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(0, len(data1), 24 / 3), dates, rotation=90)
        plt.xlabel(r"Date")
        plt.ylabel(r"Number of outages")
        plt.legend(["Real outage", "Predicted outage"], loc='upper left', fontsize=13)
        plt.title(filename)
        fig.tight_layout()
        pdf.savefig(fig)


def plot_beta_net_on_map(K, alpha, locs, filename):
    """
    References:
    - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
    - https://matplotlib.org/basemap/users/geography.html
    - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
    """

    # Make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # # rescale marker size
    # mins, maxs = 5, 300
    # dmin, dmax = data.min() if dmin is None else dmin, data.max() if dmax is None else dmax
    # print(dmin, dmax)
    # size = (data - data.min()) / (data.max() - data.min())
    # size = size * (maxs - mins) + mins
    # sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c="r")
    # # sct  = m.scatter(locs[_id, 1], locs[_id, 0], latlon=True, alpha=1., s=10, c="b")
    # # legend
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
    # plt.title(filename)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")

    thres = [ .3, .5 ]
    pairs = [ (k1, k2) for k1 in range(K) for k2 in range(K) 
        if alpha[k1, k2] > thres[0] and alpha[k1, k2] < thres[1] ]

    # spectral clustering
    from sklearn.cluster import spectral_clustering
    from scipy.sparse import csgraph
    from numpy import inf, NaN

    cmap = ["red", "yellow", "green", "blue", "black"]
    # adj  = np.zeros((model.K, model.K))
    # for k1, k2 in pairs:
    #     adj[k1, k2] = alpha[k1, k2]
    # lap  = csgraph.laplacian(alpha, normed=False)
    ls   = spectral_clustering(
        affinity=alpha,
        n_clusters=4, 
        assign_labels="kmeans",
        random_state=0)

    m.scatter(locs[:, 1], locs[:, 0], s=12, c=[ cmap[l] for l in ls ], latlon=True, alpha=0.5)

    xs = [ (locs[k1][1], locs[k2][1]) for k1, k2 in pairs ] 
    ys = [ (locs[k1][0], locs[k2][0]) for k1, k2 in pairs ]
    cs = [ alpha[k1, k2] for k1, k2 in pairs ]

    for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
        # print(i, len(cs), c, alpha.min(), alpha.max())
        w = (c - thres[0]) / (thres[1] - thres[0]) 
        m.plot(x, y, linewidth=w/2, color='grey', latlon=True, alpha=0.85)
    plt.title(filename)

    plt.savefig("imgs/%s.pdf" % filename)
        
    # plt.show()



# class AnimatedScatter(object):
#     """An animated scatter plot using matplotlib.animations.FuncAnimation."""
#     def __init__(self, numpoints=50):
#         self.locs      = np.load("data/geolocation.npy")
#         self.data      = np.load("data/maoutage.npy")[2000:4000, :]
#         # self.data      = avg(self.data, N=7) # 34589
#         print(self.data.shape)
#         self.numpoints = numpoints
#         self.stream    = self.data_stream(self.locs, self.data)

#         # Setup the figure and axes...
#         self.fig, self.ax = plt.subplots()
#         # Then setup FuncAnimation.
#         self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
#                                           init_func=self.setup_plot, blit=True)

#     def setup_plot(self):
#         """Initial drawing of the scatter plot."""
#         x, y, s, c = next(self.stream).T
#         self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=np.log(1), vmax=np.log(self.data.max()), cmap="hot")
#         self.ax.axis([self.locs[:,1].min()-0.3, self.locs[:,1].max()+0.3, self.locs[:,0].min()-0.3, self.locs[:,0].max()+0.3])
#         # For FuncAnimation's sake, we need to return the artist we'll be using
#         # Note that it expects a sequence of artists, thus the trailing comma.
#         return self.scat,

#     def data_stream(self, locs, data):
#         """
#         Generate a random walk (brownian motion). Data is scaled to produce
#         a soft "flickering" effect.
#         """
#         xy = locs
#         for t in range(data.shape[0]):
#             s = np.ones((data.shape[1])) * 10
#             c = np.log(data[t] + 1)
#             print(t)
#             yield np.c_[xy[:,1], xy[:,0], s, c]

#     def update(self, i):
#         """Update the scatter plot."""
#         data = next(self.stream)

#         # Set x and y data...
#         self.scat.set_offsets(data[:, :2])
#         # Set sizes...
#         # self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
#         # Set colors..
#         self.scat.set_array(data[:, 3])
#         # Setup title.
#         # self.ax.set_title

#         # We need to return the updated artist for FuncAnimation to draw..
#         # Note that it expects a sequence of artists, thus the trailing comma.
#         return self.scat,