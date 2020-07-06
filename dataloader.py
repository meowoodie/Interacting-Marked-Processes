#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from utils import avg, proj
from plot import plot_data_on_map

def MAdataloader():
    """
    data loader for MA data sets including outage sub data set and weather sub data set
    """
    # geolocation for weather data
    geo_weather = np.load("data/weathergeolocations.npy")
    geo_outage  = np.load("data/geolocation.npy")[:, :2]
    # from 2018-02-05 22:45:00 to 2019-01-15 04:45:00
    obs_outage = np.load("data/maoutage.npy")
    # from 2018-03-01 00:00:00 to 2018-03-31 23:00:00
    obs_wind   = np.load("data/maweather-feat008-wind.npy")
    obs_temp   = np.load("data/maweather-feat012-temp.npy")
    obs_vil    = np.load("data/maweather-feat003-VIL.npy")
    obs_gph    = np.load("data/maweather-feat011-gheight.npy")
    obs_radar  = np.load("data/maweather-feat001-radar.npy")

    # studying time window
    # from 2018-03-01 00:00:00 to 2018-03-31 23:00:00
    # outage indices from 24 * 24 * 4 + 4 = 2308 to 2308 + 15 * 24 * 4 = 3748
    # wind indices from 0 to 360
    obs_outage = avg(obs_outage[2308:3748], N=4)
    obs_wind   = obs_wind[:360]
    obs_temp   = obs_temp[:360]
    obs_vil    = obs_vil[:360]
    obs_gph    = obs_gph[:360]
    obs_radar  = obs_radar[:360]
    nzero_inds = np.nonzero(obs_wind.sum(axis=1))[0]

    # data standardization
    # # - outage
    # scl_outage = StandardScaler()
    # scl_outage.fit(obs_outage)
    # obs_outage = scl_outage.transform(obs_outage)
    # - wind
    scl_wind   = StandardScaler()
    scl_wind.fit(obs_wind)
    obs_wind   = scl_wind.transform(obs_wind)
    # - temperature
    scl_temp   = StandardScaler()
    scl_temp.fit(obs_temp)
    obs_temp   = scl_temp.transform(obs_temp)
    # - Vertically-integrated liquid 
    scl_vil    = StandardScaler()
    scl_vil.fit(obs_vil)
    obs_vil    = scl_vil.transform(obs_vil)
    # - Geopotential height
    scl_gph    = StandardScaler()
    scl_gph.fit(obs_gph)
    obs_gph    = scl_gph.transform(obs_gph)
    # - Maximum / Composite radar reflectivity
    scl_radar  = StandardScaler()
    scl_radar.fit(obs_radar)
    obs_radar  = scl_radar.transform(obs_radar)

    obs_wind   = proj(obs_wind, coord=geo_weather, proj_coord=geo_outage, k=3)
    obs_temp   = proj(obs_temp, coord=geo_weather, proj_coord=geo_outage, k=3)
    obs_vil    = proj(obs_vil, coord=geo_weather, proj_coord=geo_outage, k=3)
    obs_gph    = proj(obs_gph, coord=geo_weather, proj_coord=geo_outage, k=3)
    obs_radar  = proj(obs_radar, coord=geo_weather, proj_coord=geo_outage, k=3)


    # obs_outage = avg(obs_outage, N=8)
    # obs_temp   = avg(obs_temp, N=8)

    # obs_o = obs_outage.mean(axis=1)
    # obs_t = obs_temp.mean(axis=1)

    # # # plt.plot(obs_o[nzero_inds])
    # # # plt.plot(obs_t[nzero_inds])
    # # plt.plot(obs_o)
    # # plt.plot(obs_t)
    # # plt.show()

    # plot_data_on_map(obs_temp, geo_outage, prefix="proj-wind")

    return obs_outage, obs_temp