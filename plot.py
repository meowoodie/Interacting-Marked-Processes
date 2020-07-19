#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter


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
    handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
        func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
    plt.title(filename)
    plt.legend(handles, labels, loc="lower left", title="Num of outages")

    plt.savefig("imgs/%s.pdf" % filename)

# def plot_data_on_linechart(start_date, data, filename, dmin=None, dmax=None):
#     """
#     Args:
#     - data: [ n_timeslots ]
#     """

#     dates = [
#         "2018-03-01", "2018-03-02", "2018-03-03", "2018-03-04", "2018-03-05",
#         "2018-03-06", "2018-03-07", "2018-03-08", "2018-03-09", "2018-03-10",
#         "2018-03-11", "2018-03-12", "2018-03-13", "2018-03-14", "2018-03-15" ]

#     plt.rc('text', usetex=True)
#     font = {
#         'family' : 'serif',
#         'weight' : 'bold',
#         'size'   : 20}
#     plt.rc('font', **font)
#     with PdfPages("imgs/%s.pdf" % filename) as pdf:
#         fig, ax = plt.subplots(figsize=(12, 5))
#         ax.plot(np.arange(len(data)), data, c="#cb416b", linewidth=3, linestyle="-", label="Real", alpha=.8)
#         ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
#         plt.xticks(np.arange(0, len(data), 24 / 3), dates, rotation=90)
#         plt.xlabel(r"Date")
#         plt.ylabel(r"Number of outages")
#         # plt.legend(loc='upper left', fontsize=13)
#         plt.title(filename)
#         fig.tight_layout()
#         pdf.savefig(fig)

def plot_2data_on_linechart(start_date, data1, data2, filename, dmin=None, dmax=None, dayinterval=7):
    """
    Args:
    - data: [ n_timeslots ]
    """

    n_date = int(len(data1) / (24 * dayinterval / 3))
    dates  = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)
    with PdfPages("imgs/%s.pdf" % filename) as pdf:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.arange(len(data1)), data1, c="#677a04", linewidth=3, linestyle="--", label="Real", alpha=.8)
        ax.plot(np.arange(len(data2)), data2, c="#cea2fd", linewidth=3, linestyle="-", label="Prediction", alpha=.8)
        ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)
        plt.xticks(np.arange(0, len(data1), int(24 * dayinterval / 3)), dates, rotation=90)
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


def error_heatmap(real_data, pred_data, locs_order, start_date, dayinterval=7, modelname="Hawkes"):

    n_date = int(real_data.shape[1] / (24 * dayinterval / 3))
    dates  = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

    error_mat0  = (real_data[:, 1:] - real_data[:, :-1]) ** 2
    error_date0 = (real_data[:, 1:].mean(0) - real_data[:, :-1].mean(0)) ** 2
    error_city0 = (real_data[:, 1:].mean(1) - real_data[:, :-1].mean(1)) ** 2

    real_data  = real_data[:, 1:]
    pred_data  = pred_data[:, 1:]

    n_city     = real_data.shape[0]
    n_date     = real_data.shape[1]

    error_mat  = (real_data - pred_data) ** 2
    error_date = (real_data.mean(0) - pred_data.mean(0)) ** 2
    error_city = (real_data.mean(1) - pred_data.mean(1)) ** 2

    # cities      = [ locs[ind] for ind in locs_order ]
    city_ind    = [ 198, 315, 131, 191, 13, 43 ]
    cities      = [ "Boston", "Worcester", "Springfield", "Cambridge", "Pittsfield", "New Bedford"]

    error_mat   = error_mat[locs_order, :]
    error_mat0  = error_mat0[locs_order, :]
    error_city  = error_city[locs_order]
    error_city0 = error_city0[locs_order]

    print(error_city.argsort()[-5:][::-1])

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 12}
    plt.rc('font', **font)

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width       = 0.15, 0.65
    bottom, height    = 0.15, 0.65
    bottom_h = left_h = left + width + 0.01

    rect_imshow = [left, bottom, width, height]
    rect_date   = [left, bottom_h, width, 0.12]
    rect_city   = [left_h, bottom, 0.12, height]

    with PdfPages("%s.pdf" % modelname) as pdf:
        # start with a rectangular Figure
        fig = plt.figure(1, figsize=(8, 8))

        ax_imshow = plt.axes(rect_imshow)
        ax_city   = plt.axes(rect_city)
        ax_date   = plt.axes(rect_date)

        # no labels
        ax_city.xaxis.set_major_formatter(nullfmt)
        ax_date.yaxis.set_major_formatter(nullfmt)

        # the error matrix for cities:
        cmap = matplotlib.cm.get_cmap('magma')
        img  = ax_imshow.imshow(np.log(error_mat + 1e-5), cmap=cmap, extent=[0,n_date,0,n_city], aspect=float(n_date)/n_city)
        ax_imshow.set_yticks(city_ind)
        ax_imshow.set_yticklabels(cities, fontsize=8)
        ax_imshow.set_xticks(np.arange(0, real_data.shape[1], int(24 * dayinterval / 3)))
        ax_imshow.set_xticklabels(dates, rotation=90)
        ax_imshow.set_ylabel("City")
        ax_imshow.set_xlabel("Date")

        # the error vector for locs and dates
        ax_city.plot(error_city, np.arange(n_city), c="red", linewidth=2, linestyle="-", label="%s" % modelname, alpha=.8)
        # ax_city.plot(error_city0, np.arange(n_city), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)
        ax_date.plot(error_date, c="red", linewidth=2, linestyle="-", label="%s" % modelname, alpha=.8)
        # ax_date.plot(error_date0, c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

        ax_city.get_yaxis().set_ticks([])
        ax_city.get_xaxis().set_ticks([])
        ax_city.set_xlabel("MSE")
        ax_city.set_ylim(0, n_city)
        ax_date.get_xaxis().set_ticks([])
        ax_date.get_yaxis().set_ticks([])
        ax_date.set_ylabel("MSE")
        ax_date.set_xlim(0, n_date)
        plt.figtext(0.81, 0.133, '0')
        plt.figtext(0.91, 0.133, '%.2e' % max(max(error_city), max(error_city0)))
        plt.figtext(0.135, 0.81, '0')
        plt.figtext(0.065, 0.915, '%.2e' % max(max(error_date), max(error_date0)))
        # plt.legend()

        cbaxes = fig.add_axes([left_h, height + left + 0.01, .03, .12])
        cbaxes.get_xaxis().set_ticks([])
        cbaxes.get_yaxis().set_ticks([])
        cbaxes.patch.set_visible(False)
        cbar = fig.colorbar(img, cax=cbaxes)
        cbar.set_ticks([
            np.log(error_mat.min() + 1e-5), 
            np.log(error_mat.max() + 1e-5)
        ])
        cbar.set_ticklabels([
            0, # "%.2e" % error_mat.min(), 
            "%.2e" % error_mat.max()
        ])
        cbar.ax.set_ylabel('MSE', rotation=270, labelpad=-20)

        fig.tight_layout()
        pdf.savefig(fig)