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
from netp import TorchNetPoissonProcess



def plot_data_on_map(data, locs, prefix):
    """
    Args:
    - data: [ n_loc ]
    - locs: [ n_loc, 2 ]
    
    References:
    - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
    - https://matplotlib.org/basemap/users/geography.html
    - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
    """
    N, K = data.shape

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
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c="r")
    # legend
    handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=6, 
        func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    plt.legend(handles, labels, loc="lower left", title="Number of outages")

    plt.savefig("imgs/%s.pdf" % prefix)



# def plot_beta_net_on_map():
#     """
#     References:
#     - https://python-graph-gallery.com/315-a-world-map-of-surf-tweets/
#     - https://matplotlib.org/basemap/users/geography.html
#     - https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html
#     """
#     data     = np.load("data/maoutage.npy")[2000:4000, :]
#     N, K     = data.shape
#     spr_data = sparse.csr_matrix(data)
#     tnetp    = TorchNetPoissonProcess(d=4*6, K=K, N=N, data=spr_data)

#     tnetp.load_state_dict(torch.load("saved_models/upt-constrained-half-day.pt"))
#     beta0 = tnetp.beta0.data.numpy()[0]
#     beta1 = tnetp.beta1.data.numpy()

#     lag  = 0
#     K    = beta1.shape[0]
#     d    = beta1.shape[2]
#     locs = np.load("data/geolocation.npy")

#     plt.rc('text', usetex=True)
#     font = {
#         'family' : 'serif',
#         'weight' : 'bold',
#         'size'   : 15}
#     plt.rc('font', **font)

#     # Make the background map
#     fig = plt.figure(figsize=(8, 8))
#     ax  = plt.gca()
#     m   = Basemap(
#         projection='lcc', resolution='f', 
#         lat_0=42.1, lon_0=-71.6,
#         width=.4E6, height=.3E6)
#     m.shadedrelief()
#     m.drawcoastlines(color='gray')
#     # m.drawcountries(color='gray')
#     m.drawstates(color='gray')

#     # prepare a color for each point depending on the beta1 value.
#     sct = m.scatter(locs[:, 1], locs[:, 0], latlon=True, s=15, alpha=0.1, c="black")
#     loc_pairs = [ [(x, y), (xi, yi)] for xi, x in enumerate(locs) for yi, y in enumerate(locs) ]
#     for (x, y), (xi, yi) in tqdm(loc_pairs):
#         if beta1[xi, yi, 0] > 0.675:
#             a = [x[1], y[1]]
#             b = [x[0], y[0]]
#             print(a)
#             print(b)
#             m.plot(a, b, latlon=True, c="black")

#     # divider = make_axes_locatable(ax)
#     # cax     = divider.append_axes("right", size="5%", pad=0.1)
#     # plt.colorbar(sct, cax=cax, label=r'$\beta_1$ directed influence to blue dot')
    
#     plt.show()
#     plt.savefig("imgs/network.pdf")



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