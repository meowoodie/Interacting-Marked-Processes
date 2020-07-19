#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot import plot_data_on_map, plot_2data_on_linechart, plot_beta_net_on_map, error_heatmap

from dataloader import MAdataloader
from hkstorch import TorchHawkes

if __name__ == "__main__":

    locs    = np.load("data/geolocation.npy")
    loc_ids = locs[:, 2]
    start_date, obs_outage = MAdataloader(is_training=False)

    model = TorchHawkes(d=10, obs=obs_outage, extobs=None)

    # train(model, niter=200, lr=1., log_interval=10)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/hawkes-upt-mu-subdata.pt")

    model.load_state_dict(torch.load("saved_models/hawkes-upt-mu.pt"))

    _, lams = model()
    lams    = lams.detach().numpy()

    # ---------------------------------------------------
    #  Plot temporal predictions

    boston_ind = np.where(loc_ids == 199.)[0][0]
    worces_ind = np.where(loc_ids == 316.)[0][0]
    spring_ind = np.where(loc_ids == 132.)[0][0]
    cambri_ind = np.where(loc_ids == 192.)[0][0]
    print(boston_ind, worces_ind, spring_ind, cambri_ind)
    plot_2data_on_linechart(start_date, lams.sum(0), obs_outage.sum(0), "Total number of outages over time (testing data)", dayinterval=7)
    plot_2data_on_linechart(start_date, lams[boston_ind], obs_outage[boston_ind], "Prediction results for Boston, MA (testing data)", dayinterval=7)
    plot_2data_on_linechart(start_date, lams[worces_ind], obs_outage[worces_ind], "Prediction results for Worcester, MA (testing data)", dayinterval=7)
    plot_2data_on_linechart(start_date, lams[spring_ind], obs_outage[spring_ind], "Prediction results for Springfield, MA (testing data)", dayinterval=7)
    plot_2data_on_linechart(start_date, lams[cambri_ind], obs_outage[cambri_ind], "Prediction results for Cambridge, MA (testing data)", dayinterval=7)
    # ---------------------------------------------------



    # ---------------------------------------------------
    #  Plot error matrix

    locs_order = np.argsort(loc_ids)
    error_heatmap(real_data=obs_outage, pred_data=lams, locs_order=locs_order, start_date=start_date, dayinterval=7, modelname="ST-Cov-test")
    # ---------------------------------------------------


    

    # alpha = model.alpha.detach().numpy()
    # plot_beta_net_on_map(model.K, alpha, locs, "Correlation between locations and community structure")
    
    # plt.figure()

    # thres = [ .3, .5 ]
    # pairs = [ (k1, k2) for k1 in range(model.K) for k2 in range(model.K) 
    #     if alpha[k1, k2] > thres[0] and alpha[k1, k2] < thres[1] ]

    # # spectral clustering
    # from sklearn.cluster import spectral_clustering
    # from scipy.sparse import csgraph
    # from numpy import inf, NaN

    # cmap = ["red", "yellow", "green", "black", "purple"]
    # # adj  = np.zeros((model.K, model.K))
    # # for k1, k2 in pairs:
    # #     adj[k1, k2] = alpha[k1, k2]
    # # lap  = csgraph.laplacian(alpha, normed=False)
    # ls   = spectral_clustering(
    #     affinity=alpha,
    #     n_clusters=4, 
    #     assign_labels="kmeans",
    #     random_state=0)

    # print(len(ls))

    # plt.scatter(locs[:, 1], locs[:, 0], s=12, c=[ cmap[l] for l in ls ])

    # xs = [ (locs[k1][1], locs[k2][1]) for k1, k2 in pairs ] 
    # ys = [ (locs[k1][0], locs[k2][0]) for k1, k2 in pairs ]
    # cs = [ alpha[k1, k2] for k1, k2 in pairs ]

    # for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
    #     # print(i, len(cs), c, alpha.min(), alpha.max())
    #     w = (c - thres[0]) / (thres[1] - thres[0]) 
    #     plt.plot(x, y, linewidth=w/3, color='blue')
        
    # plt.show()



    # # ---------------------------------------------------
    # model.load_state_dict(torch.load("saved_models/hawkes-weather.pt"))
    # gamma = model.gamma.detach().numpy()
    # print(gamma.shape)
    # gamma = gamma.sum(1)
    # # plt.figure()
    # # # plt.scatter(locs[:, 1], locs[:, 0], c=gamma.sum(1), cmap="hot")
    # # for k in range(5):
    # #     plt.plot(gamma[k, :])
    # # plt.show()

    # from tqdm import tqdm
    # from scipy import sparse
    # from mpl_toolkits.basemap import Basemap
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # from matplotlib.backends.backend_pdf import PdfPages

    # plt.rc('text', usetex=True)
    # font = {
    #     'family' : 'serif',
    #     'weight' : 'bold',
    #     'size'   : 15}
    # plt.rc('font', **font)

    # # Make the background map
    # fig = plt.figure(figsize=(8, 8))
    # ax  = plt.gca()
    # m   = Basemap(
    #     projection='lcc', resolution='f', 
    #     lat_0=42.1, lon_0=-71.6,
    #     width=.4E6, height=.3E6)
    # m.shadedrelief()
    # m.drawcoastlines(color='gray')
    # m.drawstates(color='gray')

    # # # rescale marker size
    # # mins, maxs = 5, 100
    # # dmin, dmax = gamma.min(), gamma.max()
    # # size = (gamma - dmin) / (dmax - dmin)
    # # size = size * (maxs - mins) + mins
    # # sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c="blue")
    # # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    # #     func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)
    # # plt.title("Coefficients of temperature")
    # # plt.legend(handles, labels, loc="lower left", title="Value of coefficients")

    # # plt.savefig("imgs/Coefficients of temperature.pdf")

    # # rescale marker size
    # dmin, dmax = gamma.min(), gamma.max()
    # cmin, cmax = - max(abs(dmin), abs(dmax)), max(abs(dmin), abs(dmax))
    # print(cmin, cmax)
    # cm  = plt.cm.get_cmap('bwr')
    # sct = m.scatter(locs[:, 1], locs[:, 0], c=gamma, latlon=True, alpha=0.5, s=12, cmap=cm, vmin=cmin, vmax=cmax)
    # # fig.colorbar(sct, cax=ax)
    # plt.title("Coefficients of temperature")
    # plt.colorbar(sct, ax=ax, fraction=0.05, pad=0.05)
    # plt.savefig("imgs/Coefficients of temperature.pdf")