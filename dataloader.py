#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import arrow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from utils import avg, proj #, scale_down_data

feat_list = [
    "001", "002", "003", "004", "005",
    "006", "007", "008", "009", "010",
    "011", "012", "013", "014", "015",
    "016", "017"
]

def MAdataloader(is_training=True):
    """
    data loader for MA data sets including outage sub data set and weather sub data set
    """
    # geolocation for weather data
    geo_weather = np.load("data/weathergeolocations.npy")
    geo_outage  = np.load("data/geolocation_new.npy")[:, :2]
    # from 2018-02-05 22:45:00 to 2019-01-15 04:45:00
    obs_outage = np.load("data/maoutage_new.npy")
    # from 2018-03-01 00:00:00 to 2018-03-31 23:00:00
    obs_feats  = [ np.load("data/maweather-feat%s.npy" % feat) for feat in feat_list ]

    if is_training:
        # training time window
        # from 2018-03-01 00:00:00 to 2018-03-31 23:00:00
        # outage indices from 24 * 24 * 4 + 4 = 2308 to 2308 + 15 * 24 * 4 = 3748
        # wind indices from 0 to 360
        obs_outage = avg(obs_outage[2308:3748], N=4)
        obs_feats  = [ obs[:360] for obs in obs_feats ]
        # nzero_inds = np.nonzero(obs_wind.sum(axis=1))[0]
        start_date = arrow.get(datetime(2018, 3, 1))
    else:
        # testing time window
        # from 2018-09-05 00:00:00 to 2018-11-19 00:00:00
        # (24 + 188) * 24 * 4 + 4 = 20356 to (24 + 263) * 24 * 4 + 4 = 27556
        obs_outage = avg(obs_outage[20356:27556], N=4)
        # TODO: make testing weather data available soon
        obs_feats  = [ obs[:360] for obs in obs_feats ]
        # nzero_inds = np.nonzero(obs_wind.sum(axis=1))[0]
        start_date = arrow.get(datetime(2018, 9, 5))

    # data standardization
    _obs_feats = []
    for obs in obs_feats:
        scl = StandardScaler()
        scl.fit(obs)
        obs = scl.transform(obs)
        _obs_feats.append(obs)
    obs_feats = _obs_feats

    # project weather data to the coordinate system that outage data is using
    obs_feats = [ proj(obs, coord=geo_weather, proj_coord=geo_outage, k=10) for obs in obs_feats ] 

    obs_outage = avg(obs_outage, N=3).transpose()
    obs_feats   = [ avg(obs, N=3).transpose() for obs in obs_feats ]
    obs_weather = np.stack(obs_feats, axis=2) 
    # obs_weather = (obs_weather - obs_weather.min()) / (obs_weather.max() - obs_weather.min())

    return start_date, obs_outage, obs_weather