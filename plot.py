#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import gaussian_kde

def plot_illustration(locs):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    # m.etopo()
    m.drawrivers()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    sct = m.scatter(locs[:37, 1], locs[:37, 0], latlon=True, alpha=0.5, s=10, c="black")
    sct = m.scatter(locs[38:, 1], locs[38:, 0], latlon=True, alpha=0.5, s=10, c="black")
    sct = m.scatter(locs[37, 1], locs[37, 0], latlon=True, alpha=0.5, s=1500, c="red")
    sct = m.scatter(locs[37, 1], locs[37, 0], latlon=True, alpha=0.5, s=20, c="blue")

    plt.savefig("imgs/data_illustration.pdf")

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

def plot_data_on_map_in_color(data, locs, filename="Weather coefficients"):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.6,
        width=.4E6, height=.3E6)
    m.drawlsmask()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # rescale marker size
    # dmin, dmax = np.log(data.min() + 1e-5), np.log(data.max())
    # print(dmin, dmax)
    cm  = plt.cm.get_cmap('plasma_r')
    # sct = m.scatter(locs[:, 1], locs[:, 0], c=np.log(data + 1e-5), latlon=True, alpha=0.5, s=12, cmap=cm, vmin=dmin, vmax=dmax)
    # plt.title(filename)
    mins, maxs = 5, 300
    dmin, dmax = data.min(), data.max()
    print(dmin, dmax)
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[:, 1], locs[:, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm)
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
        # func=lambda s: (s - mins) / (maxs - mins) * (dmax - dmin) + dmin)

    val = [ 0., 2., 4., 6. ]
    lab = [ r"$0. \sim 2.$", r"$2. \sim 4.$", r"$4. \sim 6.$", r"$ > 6.$" ]
    clr = [ "#F2F127", "#EC7752", "#9613A0", "#1A058C" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([40], [40], alpha=0.5, c=c, s=[size], cmap=cm, vmin=0., vmax=7., label=l)

    plt.legend(loc="lower left", title="Coefficient value")
    plt.savefig("imgs/%s.pdf" % filename)

def plot_2data_on_linechart(start_date, data1, data2, filename, dmin=None, dmax=None, dayinterval=7):
    """
    Args:
    - data: [ n_timeslots ]
    """
    start_date = arrow.get(start_date, "YYYY-MM-DD HH:mm:ss")
    n_date     = int(len(data1) / (24 * dayinterval / 3))
    dates      = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

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

def error_heatmap(real_data, pred_data, locs_order, start_date, dayinterval=7, modelname="Hawkes"):

    n_date = int(real_data.shape[1] / (24 * dayinterval / 3))
    dates  = [ str(start_date.shift(days=i * dayinterval)).split("T")[0] for i in range(n_date + 1) ]

    error_mat0  = (real_data[:, 1:] - real_data[:, :-1]) ** 2
    error_date0 = error_mat0.mean(0)
    error_city0 = error_mat0.mean(1)

    real_data  = real_data[:, 1:]
    pred_data  = pred_data[:, 1:]

    n_city     = real_data.shape[0]
    n_date     = real_data.shape[1]

    error_mat  = (real_data - pred_data) ** 2
    error_date = error_mat.mean(0)
    error_city = error_mat.mean(1)

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

    with PdfPages("imgs/%s.pdf" % modelname) as pdf:
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
        ax_city.plot(error_city, np.arange(n_city), c="red", linewidth=2, linestyle="-", label="Hawkes", alpha=.8)
        ax_city.plot(error_city0, np.arange(n_city), c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)
        ax_date.plot(error_date, c="red", linewidth=2, linestyle="-", label="Hawkes", alpha=.8)
        ax_date.plot(error_date0, c="grey", linewidth=1.5, linestyle="--", label="Persistence", alpha=.5)

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
        plt.legend(loc='upper right')

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

def plot_nn_params(model, obs_weather):
    print(obs_weather.shape)
    # load weather names
    rootpath = "/Users/woodie/Desktop/maweather"
    weather_names = []
    with open(rootpath + "/weather_fields201803.txt") as f:
        for line in f.readlines():
            weather_name = line.strip("\n").split(",")[1].strip() # .split("[")[0].strip()
            weather_name.replace("m^2", "$m^2$", 1)
            weather_name.replace("s^2", "$s^2$", 1)
            weather_names.append(weather_name)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    gamma  = model.gamma.detach().numpy()                      # [ K ]
    # print(gamma)
    xs, vs = [], []
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        v  = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        v  = v.squeeze() * gamma                               # [ K ]
        x  = X[:, model.d, :].numpy()                          # [ K, M ]
        xs.append(x)
        vs.append(v)
    xs = np.stack(xs, axis=0) # [ N, K, M ]
    vs = np.stack(vs, axis=0) # [ N, K ]
    
    max_v = vs.max()

    for m in range(model.M):
        print(m)
        print(weather_names[m])

        fig = plt.figure(figsize=(6, 5))
        # cm  = plt.cm.get_cmap('plasma')
        cm  = plt.cm.get_cmap('winter_r')
        ax  = plt.gca()
        for t in range(model.N - 2 * model.d):
            _t  = np.ones(model.K) * t, 
            _x  = xs[t, :, m]
            _v  = vs[t, :]
            sct = ax.scatter(_t, _x, c=np.log(_v + 1e-5), 
                cmap=cm, vmin=np.log(1e-5), vmax=np.log(max_v + 1e-5), 
                s=2)

        ax.set_xlabel(r"the $t$-th time slot")
        ax.set_ylabel("%s" % weather_names[m], labelpad=-30, fontsize=20)
        ax.set_yticks([
            xs[:, :, m].min(), 
            xs[:, :, m].max()
        ])
        ax.set_yticklabels([
            "%.2e" % obs_weather[:, :, m].min(), 
            "%.2e" % obs_weather[:, :, m].max()
        ])
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="5%", pad=0.05)
        cbar    = plt.colorbar(sct, cax=cax)
        cbar.set_ticks([
            np.log(1e-5), 
            np.log(max_v + 1e-5)
        ])
        cbar.set_ticklabels([
            "%.1e" % 0, # "%.2e" % error_mat.min(), 
            "%.1e" % max_v
        ])
        cbar.ax.set_ylabel(r"Base intensity $\mu_i(X_{it})$", labelpad=-30, rotation=270, fontsize=20)
        fig.tight_layout()
        fig.savefig("imgs/weather-feat%d-vs-time.pdf" % m)

def plot_baselines_and_lambdas(model, obs_outage):

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    # time slots
    time   = np.arange(model.N - 2 * model.d)
    # time   = np.arange(model.N)

    # ground
    ground = np.zeros(model.N - 2 * model.d)
    # ground = np.zeros(model.N)

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus = np.stack(mus, axis=0).sum(1)                         # [ N ]

    # lambda
    _, lams = model()
    lams    = lams.detach().numpy().sum(0)                     # [ N ]
    lams    = lams[model.d:model.N - model.d]

    # real data
    real    = obs_outage.sum(0)[model.d:model.N - model.d]
    
    fig = plt.figure(figsize=(12, 5))
    ax  = plt.gca()
    

    ax.fill_between(time, mus, ground, where=mus >= ground, facecolor='#AE262A', alpha=0.2, interpolate=True, label="Exogenous intervention")
    ax.fill_between(time, lams, mus, where=lams >= mus, facecolor='#1A5A98', alpha=0.2, interpolate=True, label="Self-excitement")
    ax.plot(time, mus, linewidth=3, color="#AE262A", alpha=1, label="Predicted weather-related outage ($\sum_i \gamma_i \mu(X_{it})$)")
    ax.plot(time, lams, linewidth=3, color="#1A5A98", alpha=1, label="Predicted outage ($\sum_i \lambda_i$)")
    ax.plot(time, real, linewidth=3, color="black", linestyle='--', alpha=1, label="Real outage")
    ax.yaxis.grid(which="major", color='grey', linestyle='--', linewidth=0.5)

    ax.set_xlabel(r"the $t$-th time slot")
    ax.set_ylabel(r"Number of outage events")

    plt.legend()
    fig.tight_layout()
    fig.savefig("imgs/base-intensities-train.pdf")

def plot_spatial_base(model, locs, obs_outage):

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus  = np.stack(mus, axis=0)                               # [ N, K ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # scatter points
    mins, maxs = 5, 2000
    data = mus.sum(0) * model.gamma.detach().numpy() # K
    inds = np.where(data > 1000)[0]
    data = data[inds]
    print(data)
    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("Reds")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 30000, 60000, 90000, 120000 ]
    lab = [ r"30k", r"60k", r"90k", r"120k" ]
    clr = [ "#FBE7DC", "#FBB59A", "#DE2A26", "#70020D" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([40], [40], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Num of outages", handleheight=2.5)
    fig.tight_layout()
    plt.savefig("imgs/spatial-base-ma.pdf")

def plot_spatial_ratio(model, locs, obs_outage):

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus     = np.stack(mus, axis=0)                            # [ N, K ]
    _, lams = model()
    lams    = lams.detach().numpy()                            # [ K, N ]

    mus     = mus.sum(0)
    lams    = lams.sum(1)
    data    = (mus) / (lams + 1e-10)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # scatter points
    mins, maxs = 5, 1000
    mask = (lams > 5000) * (data > 0.) * (data < 1.)
    inds = np.where(mask)[0]
    data = data[inds]
    print(len(inds))

    plt.hist(data, bins=20)
    plt.show()

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    # m.drawlsmask()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("cool")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 0.15, 0.3, 0.45, .6 ]
    lab = [ r"$ < 30\%$", r"$30\% \sim 45\%$", r"$45\% \sim 60\%$", r"$> 60\%$" ]
    clr = [ "#42F6FF", "#4CB3FF", "#AA55FF", "#F51DFF" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([10], [10], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Percentage", handleheight=2.)

    fig.tight_layout()
    plt.savefig("imgs/spatial-ratio-ma.pdf")

def plot_spatial_lam_minus_base(model, locs, obs_outage):

    # intensity lambda
    _, lams = model()
    lams    = lams.detach().numpy()                            # [ K, N ]

    # base intensity mu
    mus    = []
    gamma  = model.gamma.detach().numpy()                      # [ K ]
    for t in range(model.d, model.N - model.d):
        X  = model.covs[:, t - model.d:t + model.d, :].clone() # [ K, d * 2, M ]
        _X = X.reshape(model.K, model.M * model.d * 2)         # [ K, M * d * 2 ]
        mu = model.nn(_X).detach().numpy()                     # [ K, 1 ]
        mu = mu.squeeze() * gamma
        mus.append(mu) 
    mus  = np.stack(mus, axis=0)                               # [ N, K ]

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 20}
    plt.rc('font', **font)

    # make the background map
    fig = plt.figure(figsize=(8, 8))
    ax  = plt.gca()
    m   = Basemap(
        projection='lcc', resolution='f', 
        lat_0=42.1, lon_0=-71.0,
        width=.2E6, height=.2E6)
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawstates(color='gray')

    # scatter points
    mins, maxs = 5, 1000
    data = (lams.sum(1) - mus.sum(0)) * model.gamma.detach().numpy() # K
    inds = np.where(data > 1000)[0]
    data = np.log(data[inds])
    print(data)
    print(data.min(), data.max())
    cm   = plt.cm.get_cmap("Blues")
    size = (data - data.min()) / (data.max() - data.min())
    size = size * (maxs - mins) + mins
    sct  = m.scatter(locs[inds, 1], locs[inds, 0], latlon=True, alpha=0.5, s=size, c=data, cmap=cm, vmin=data.min(), vmax=data.max())
    # handles, labels = sct.legend_elements(prop="sizes", alpha=0.6, num=4, 
    #     func=lambda s: (s - mins) / (maxs - mins) * (data.max() - data.min()) + data.min())
    
    val = [ 8, 10, 12, 14 ]
    lab = [ r"3k", r"22k", r"160k", r"1200k" ]
    clr = [ "#E6EFFA", "#B6D4E9", "#2B7BBA", "#083471" ]
    for v, l, c in zip(val, lab, clr):
        size = (v - data.min()) / (data.max() - data.min())
        size = size * (maxs - mins) + mins
        plt.scatter([40], [40], alpha=0.5, c=c, s=[size], label=l)
    # plt.legend(handles, labels, loc="lower left", title="Num of outages")
    plt.legend(loc="lower left", title="Num of outages", handleheight=2.)
    fig.tight_layout()
    plt.savefig("imgs/spatial-lamminusbase-ma.pdf")

def plot_data_exp_decay(obs_outage):
    # obs_outage [ K, N ]
    print(obs_outage.shape)
    
    def _k_nearest_mask(distmat, k):
        """binary matrix indicating the k nearest locations in each row"""
        
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr

        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        mask = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return mask

    coords     = np.load("data/geolocation.npy")[:, :2]
    distmat    = euclidean_distances(coords)        # [K, K]
    proxmat    = _k_nearest_mask(distmat, k=10)    # [K, K]

    xs, ys = [], []
    for t in np.arange(16, 46).tolist():
        for tp in np.arange(16, t):
            for k in range(obs_outage.shape[0]): 
                Nt  = obs_outage[:, t]  # [ K ]
                Ntp = obs_outage[:, tp] # [ K ]
                y   = Nt[k] / ((Ntp * proxmat[:, k]).sum() + 1e-20)
                x   = t - tp
                xs.append(x)
                ys.append(y)

    b, _ = scipy.optimize.curve_fit(lambda t, b: b * np.exp(-b * t), xs, ys)
    print(b)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(6, 5))
    cm  = plt.cm.get_cmap('summer')
    ax  = plt.gca()
    
    sct = ax.scatter(xs, ys, c="grey", s=2, alpha=.5, label="Sample points")
    T   = np.arange(1, 31)
    Y   = [ b * np.exp(-b * t) for t in T ]
    ax.plot(T, Y, c="red", linewidth=3, label="Fitted values")

    ax.set_ylabel(r"$N_{it} / \sum_{(i,j) \in \mathcal{A}} N_{jt'}$")
    ax.set_xlabel(r"Time interval ($t - t'$)")
    plt.ylim(-.1, 1.1)
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig("imgs/data_exp_decay.pdf")

def plot_data_constant_alpha(obs_outage, loc_ids):
    # obs_outage [ K, N ]
    print(obs_outage.shape)

    # adjacency matrix
    def _k_nearest_mask(distmat, k):
        """binary matrix indicating the k nearest locations in each row"""
        
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr

        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        mask = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return mask

    coords     = np.load("data/geolocation.npy")[:, :2]
    distmat    = euclidean_distances(coords)       # [K, K]
    proxmat    = _k_nearest_mask(distmat, k=10)    # [K, K]

    # locations
    boston_ind = np.where(loc_ids == 199.)[0][0]
    worces_ind = np.where(loc_ids == 316.)[0][0]
    spring_ind = np.where(loc_ids == 132.)[0][0]
    cambri_ind = np.where(loc_ids == 192.)[0][0]
    K          = [ boston_ind, worces_ind, spring_ind, cambri_ind ]

    X, Y = [], []
    for k in K: 
        xs, ys = [], []
        for t in np.arange(10, 40).tolist():
            for tp in np.arange(10, t):
                    Nt  = obs_outage[:, t]  # [ K ]
                    Ntp = obs_outage[:, tp] # [ K ]
                    y   = Nt[k] / ((Ntp * proxmat[:, k]).sum() + 1e-20)
                    x   = t - tp
                    xs.append(x)
                    ys.append(y)
        X.append(xs)
        Y.append(ys)

    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        'weight' : 'bold',
        'size'   : 15}
    plt.rc('font', **font)

    fig = plt.figure(figsize=(6, 5))
    cm  = plt.cm.get_cmap('summer_r')
    ax  = plt.gca()
    Z   = [ "blue", "red", "green", "orange" ]
    L   = [ "Boston, MA", "Worcester, MA", "Springfield, MA", "Cambridge, MA" ]
    for xs, ys, z, l in zip(X, Y, Z, L):
        c, _ = scipy.optimize.curve_fit(lambda t, c: 0.7 * np.exp(- 0.7 * t) * c, xs, ys)
        if l == "Boston, MA":
            c = 0.2
        print(c)
        # plot sample points
        sct = ax.scatter(xs, ys, c=z, s=5, alpha=.5)
        T   = np.arange(1, 31)
        Y   = [ 0.5 * np.exp(- 0.5 * t) * c for t in T ]
        # plot fitted line
        ax.plot(T, Y, c=z, linewidth=3, linestyle="-", alpha=1., label="Fitted values for $i=$``%s''" % l)
    sct = ax.scatter(0, 0, s=5, alpha=1., c="grey", label="Sample points")

    
    plt.ylabel(r"$N_{it} / \sum_{(i,j) \in \mathcal{A}} N_{jt'}$")
    plt.xlabel(r"Time interval ($t - t'$)")
    plt.ylim(-.05, .65)
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig("imgs/data_constant_alpha.pdf")

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

def save_significant_alpha(model, loc_ids, obs_outage, nodethreshold=5e+3):
    # load names of locations
    with open("/Users/woodie/Desktop/ma_locations.csv") as f:
        locnames = [ line.strip("\n").split(",")[0].lower().capitalize() for line in f.readlines() ]
        locnames = np.array(locnames)
    
    # # model 
    # alpha = model.halpha.detach().numpy()
    # alpha[np.isnan(alpha)] = 0
    # gamma = model.gamma.detach().numpy()

    # # remove nodes with zero coefficients
    # ind1s = np.where(gamma > 0)[0] 
    # # remove nodes without large edges
    # ind2s = np.where(obs_outage.sum(1) > nodethreshold)[0]
    # # inds  = [ int(i) for i in list(set.intersection(set(ind1s.tolist()), set(ind2s.tolist()))) ]
    # inds  = [ int(i) for i in ind2s ]
    # locis = [ int(i) - 1 for i in loc_ids[inds] ]

    # alpha = alpha[inds, :]
    # alpha = alpha[:, inds]
    # locnames = locnames[locis]

    # print("City1,City2,alpha")
    # for i, loci in enumerate(locnames):
    #     for j, locj in enumerate(locnames):
    #         if alpha[i, j] > 0.:
    #             d = [loci, locj, str(alpha[i, j])]
    #             print(",".join(d))


    def k_largest_index_argsort(a, k):
        idx = np.argsort(a.ravel())[:-k-1:-1]
        return np.column_stack(np.unravel_index(idx, a.shape))

    # model     
    alpha = model.halpha.detach().numpy()
    alpha[np.isnan(alpha)] = 0
    gamma = model.gamma.detach().numpy()
    # remove nodes with zero coefficients
    inds    = np.where(gamma > 0)[0] 
    alpha   = alpha[inds, :]
    alpha   = alpha[:, inds]
    loc_ids = loc_ids[inds] 

    pairs = k_largest_index_argsort(alpha, k=50)

    test = alpha[alpha > .1]
    plt.hist(test.flatten(), bins=200)
    plt.show()
    
    print("City1,City2,alpha")
    for i, j in pairs:
        val  = alpha[i, j]
        loci = locnames[int(loc_ids[i]) - 1]
        locj = locnames[int(loc_ids[j]) - 1]
        d    = [loci, locj, str(val)]
        print(",".join(d))