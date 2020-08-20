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

config = {
    "MA Mar 2018": {
        # outage configurations
        "outage_path":    "maoutage_2018.npy",
        "outage_startt":  "2017-12-31 00:00:00",
        "outage_endt":    "2019-01-15 04:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201803",
        "weather_startt": "2018-03-01 00:00:00",
        "weather_endt":   "2018-03-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      [
            "001", "002", "003", "004", "005",
            "006", "007", "008", "009", "010",
            "011", "012", "013", "014", "015",
            "016", "017"],
        # time window
        "_startt":        "2018-03-01 00:00:00",
        "_endt":          "2018-03-17 00:00:00"
    },
    "MA Oct 2018": {
        # outage configurations
        "outage_path":    "maoutage_2018.npy",
        "outage_startt":  "2017-12-31 00:00:00",
        "outage_endt":    "2019-01-15 04:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201810",
        "weather_startt": "2018-10-01 00:00:00",
        "weather_endt":   "2018-10-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      [
            "001", "002", "003", "004", "005",
            "006", "007", "008", "009", "010",
            "011", "012", "013", "014", "015",
            "016", "017"],
        # time window
        "_startt":        "2018-10-01 00:00:00",
        "_endt":          "2018-10-31 00:00:00"
    },
    "MA Feb 2019": {
        # outage configurations
        "outage_path":    "maoutage_2019.npy",
        "outage_startt":  "2019-01-01 00:00:00",
        "outage_endt":    "2019-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201902",
        "weather_startt": "2019-02-01 00:00:00",
        "weather_endt":   "2019-02-28 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      [
            "001", "002", "003", "004", "005",
            "006", "007", "008", "009", "010",
            "011", "012", "013", "014", "015",
            "016", "017"],
        # time window
        "_startt":        "2019-02-11 00:00:00",
        "_endt":          "2019-02-28 00:00:00"
    },
    "MA Oct 2019": {
        # outage configurations
        "outage_path":    "maoutage_2019.npy",
        "outage_startt":  "2019-01-01 00:00:00",
        "outage_endt":    "2019-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201910",
        "weather_startt": "2019-10-01 00:00:00",
        "weather_endt":   "2019-10-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      [
            "001", "002", "003", "004", "005",
            "006", "007", "008", "009", "010",
            "011", "012", "013", "014", "015",
            "016", "017"],
        # time window
        "_startt":        "2019-10-01 00:00:00",
        "_endt":          "2019-10-31 00:00:00"
    }
}



def load_outage(config, N=4):
    # load geo locations appeared in outage data
    geo_outage = np.load("data/geolocation_351.npy")
    # load outage data
    print("[%s] reading outage data from data/%s ..." % (arrow.now(), config["outage_path"]))
    obs_outage = np.load("data/%s" % config["outage_path"])
    print("[%s] outage data with shape %s are loaded." % (arrow.now(), obs_outage.shape))

    # check if the start date and end date of outage data
    freq       = config["outage_freq"]
    startt     = arrow.get(config["outage_startt"], "YYYY-MM-DD HH:mm:ss")
    endt       = arrow.get(config["outage_endt"], "YYYY-MM-DD HH:mm:ss")
    assert int((endt.timestamp - startt.timestamp) / freq + 1) == obs_outage.shape[0], "incorrect number of recordings or incorrect dates."

    # select data in the time window
    start_date = arrow.get(config["_startt"], "YYYY-MM-DD HH:mm:ss")
    end_date   = arrow.get(config["_endt"], "YYYY-MM-DD HH:mm:ss")
    startind   = int((start_date.timestamp - startt.timestamp) / freq)
    endind     = int((end_date.timestamp - startt.timestamp) / freq)
    obs_outage = obs_outage[startind:endind+1, :] # [ n_times, n_locations ]
    print("[%s] outage data with shape %s are extracted, from %s (ind: %d) to %s (ind: %d)" % \
        (arrow.now(), obs_outage.shape, start_date, startind, end_date, endind))

    # rescale outage data
    obs_outage = avg(obs_outage, N=N)

    return obs_outage, geo_outage



def load_weather(config):
    # load geo locations appeared in weather data
    geo_weather = np.load("data/weathergeolocations.npy")
    # load outage data
    print("[%s] reading weather data from data/%s ..." % (arrow.now(), config["weather_path"]))
    obs_feats  = [ np.load("data/%s/%s-feat%s.npy" % (config["weather_path"], config["weather_path"], feat)) for feat in config["feat_list"] ]
    obs_feats  = np.stack(obs_feats, 0)
    print("[%s] weather data with shape %s are loaded." % (arrow.now(), obs_feats.shape))

    # check if the start date and end date of weather data
    freq       = config["weather_freq"]
    startt     = arrow.get(config["weather_startt"], "YYYY-MM-DD HH:mm:ss")
    endt       = arrow.get(config["weather_endt"], "YYYY-MM-DD HH:mm:ss")
    assert int((endt.timestamp - startt.timestamp) / freq + 1) == obs_feats.shape[1], "incorrect number of recordings or incorrect dates."

    # select data in the time window
    start_date = arrow.get(config["_startt"], "YYYY-MM-DD HH:mm:ss")
    end_date   = arrow.get(config["_endt"], "YYYY-MM-DD HH:mm:ss")
    startind   = int((start_date.timestamp - startt.timestamp) / freq)
    endind     = int((end_date.timestamp - startt.timestamp) / freq)
    obs_feats  = obs_feats[:, startind:endind+1, :] # [ n_feats, n_times, n_locations ]
    print("[%s] weather data with shape %s are extracted, from %s (ind: %d) to %s (ind: %d)" % \
        (arrow.now(), obs_feats.shape, start_date, startind, end_date, endind))

    return obs_feats, geo_weather


def dataloader(config, standardization=True):
    """
    data loader for MA data sets including outage sub data set and weather sub data set

    - season: summer or winter
    """
    obs_outage, geo_outage = load_outage(config)
    obs_feats, geo_weather = load_weather(config)

    # data standardization
    print("[%s] weather data standardization ..." % arrow.now())
    if standardization:
        _obs_feats = []
        for obs in obs_feats:
            scl = StandardScaler()
            scl.fit(obs)
            obs = scl.transform(obs)
            _obs_feats.append(obs)
        obs_feats = _obs_feats

    # project weather data to the coordinate system that outage data is using
    print("[%s] weather data projection ..." % arrow.now())
    obs_feats   = [ proj(obs, coord=geo_weather, proj_coord=geo_outage[:, :2], k=10) for obs in obs_feats ]

    obs_outage  = avg(obs_outage, N=3).transpose()                   # [ n_locations, n_times ]
    obs_feats   = [ avg(obs, N=3).transpose() for obs in obs_feats ] # ( n_feats, [ n_locations, n_times ] )
    obs_weather = np.stack(obs_feats, 2)                             # [ n_locations, n_times, n_feats ]

    # outage_show  = (obs_outage.sum(0) - obs_outage.sum(0).min()) / (obs_outage.sum(0).max() - obs_outage.sum(0).min())
    # weather_show = (obs_weather[:, :, 0].sum(0) - obs_weather[:, :, 0].sum(0).min()) / (obs_weather[:, :, 0].sum(0).max() - obs_weather[:, :, 0].sum(0).min())
    # plt.plot(outage_show)
    # plt.plot(weather_show)
    # plt.show()

    return obs_outage, obs_weather, geo_outage

if __name__ == "__main__":
    dataloader(config["MA Oct 2019"], standardization=False)