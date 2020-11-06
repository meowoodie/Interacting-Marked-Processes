#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import arrow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from utils import avg, proj #, scale_down_data

# new_feat_list = [
#     "001", "002", "003", "004", "005", "006", "007", "008", "028", "029", "030", "031", "032", "033", "034", "035", 
#     "036", "037", #
#     "038", "039", "040", "041", "055", "056", "057", "059", "061", 
#     "062", "063", # 
#     "064", "065", "066", "068", "071", "072", "073", 
#     "088", "093", #
#     "094", 
#     "095", "097", #
#     "117", "118"
# ]

# old_feat_list = [
#     "001", "002", "003", "004", "005", "006", "007", "008", "026", "027", "028", "029", "030", "031", "032", "033", 
#     "034", "035", #
#     "036", "037", "038", "039", "043", "044", "045", "047", "049", 
#     "050", "051", #
#     "052", "053", "054", "056", "059", "060", "061", 
#     "073", "078", # 
#     "079", 
#     "080", "082", #
#     "101", "102"
# ]

concise_new_feat_list = [
    "001", "002", "003", "004", "005", "006", "007", "008", "028", "029", 
    "030", "031", "032", "033", "034", "035", "038", "039", "040", "041", 
    "055", "056", "057", "059", "061", "064", "065", "066", "068", "071", 
    "072", "073", "094", "117", "118"
]

concise_old_feat_list = [
    "001", "002", "003", "004", "005", "006", "007", "008", "026", "027", 
    "028", "029", "030", "031", "032", "033", "036", "037", "038", "039", 
    "043", "044", "045", "047", "049", "052", "053", "054", "056", "059", 
    "060", "061", "079", "101", "102"
]

config = {
    "MA Mar 2018": {
        # outage configurations
        "outage_path":    "maoutage_2018.npy",
        "outage_geo":     "geolocation_351.npy",
        "outage_startt":  "2017-12-31 00:00:00",
        "outage_endt":    "2019-01-15 04:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201803",
        "weather_geo":    "ma_weathergeolocations.npy",
        "weather_startt": "2018-03-01 00:00:00",
        "weather_endt":   "2018-03-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_old_feat_list,
        # time window
        "_startt":        "2018-03-01 00:00:00",
        "_endt":          "2018-03-16 00:00:00" # "2018-03-17 00:00:00"
    },
    "Normal MA Mar 2018": {
        # outage configurations
        "outage_path":    "maoutage_2018.npy",
        "outage_geo":     "geolocation_351.npy",
        "outage_startt":  "2017-12-31 00:00:00",
        "outage_endt":    "2019-01-15 04:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201803",
        "weather_geo":    "ma_weathergeolocations.npy",
        "weather_startt": "2018-03-01 00:00:00",
        "weather_endt":   "2018-03-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_old_feat_list,
        # time window
        "_startt":        "2018-03-01 00:00:00",
        "_endt":          "2018-03-31 00:00:00"
    },
    "MA Oct 2018": {
        # outage configurations
        "outage_path":    "maoutage_2018.npy",
        "outage_geo":     "geolocation_351.npy",
        "outage_startt":  "2017-12-31 00:00:00",
        "outage_endt":    "2019-01-15 04:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201810",
        "weather_geo":    "ma_weathergeolocations.npy",
        "weather_startt": "2018-10-01 00:00:00",
        "weather_endt":   "2018-10-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2018-10-01 00:00:00",
        "_endt":          "2018-10-31 00:00:00"
    },
    "MA Feb 2019": {
        # outage configurations
        "outage_path":    "maoutage_2019.npy",
        "outage_geo":     "geolocation_351.npy",
        "outage_startt":  "2019-01-01 00:00:00",
        "outage_endt":    "2019-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201902",
        "weather_geo":    "ma_weathergeolocations.npy",
        "weather_startt": "2019-02-01 00:00:00",
        "weather_endt":   "2019-02-28 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_old_feat_list,
        # time window
        "_startt":        "2019-02-11 00:00:00",
        "_endt":          "2019-02-28 00:00:00"
    },
    "MA Oct 2019": {
        # outage configurations
        "outage_path":    "maoutage_2019.npy",
        "outage_geo":     "geolocation_351.npy",
        "outage_startt":  "2019-01-01 00:00:00",
        "outage_endt":    "2019-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "maweather-201910",
        "weather_geo":    "ma_weathergeolocations.npy",
        "weather_startt": "2019-10-01 00:00:00",
        "weather_endt":   "2019-10-31 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2019-10-01 00:00:00",
        "_endt":          "2019-10-31 00:00:00"
    },
    "Complete GA Oct 2018": {
        # outage configurations
        "outage_path":    "gaoutage_201809-11.npy",
        "outage_geo":     "ga_geolocation_665.npy",
        "outage_startt":  "2018-09-13 00:00:00",
        "outage_endt":    "2018-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_gaweather-20180911",
        "weather_geo":    "ga_geolocation_665.npy",
        "weather_startt": "2018-09-12 00:00:00",
        "weather_endt":   "2018-11-30 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2018-10-05 00:00:00",
        "_endt":          "2018-11-05 00:00:00"
    },
    "GA Oct 2018": {
        # outage configurations
        "outage_path":    "gaoutage_201809-11.npy",
        "outage_geo":     "ga_geolocation_665.npy",
        "outage_startt":  "2018-09-13 00:00:00",
        "outage_endt":    "2018-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_gaweather-20180911",
        "weather_geo":    "ga_geolocation_665.npy",
        "weather_startt": "2018-09-12 00:00:00",
        "weather_endt":   "2018-11-30 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2018-10-05 00:00:00",
        "_endt":          "2018-10-20 00:00:00"
    },
    "NCSC Summer 2020": {
        # outage configurations
        "outage_path":    "ncoutage_202005-09.npy",
        "outage_geo":     "nc_geolocation_115.npy",
        "outage_startt":  "2020-05-01 00:00:00",
        "outage_endt":    "2020-09-14 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_ncscweather-202005",
        "weather_geo":    "nc_geolocation_115.npy",
        "weather_startt": "2020-05-01 00:00:00",
        "weather_endt":   "2020-09-14 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2020-05-01 00:00:00",
        "_endt":          "2020-09-14 00:00:00"
    },
    "NCSC May 2020": {
        # outage configurations
        "outage_path":    "ncoutage_202005-09.npy",
        "outage_geo":     "nc_geolocation_115.npy",
        "outage_startt":  "2020-05-01 00:00:00",
        "outage_endt":    "2020-09-14 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_ncscweather-202005",
        "weather_geo":    "nc_geolocation_115.npy",
        "weather_startt": "2020-05-01 00:00:00",
        "weather_endt":   "2020-09-14 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        # Tropical Storm Arthur
        "_startt":        "2020-05-15 00:00:00",
        "_endt":          "2020-06-01 00:00:00"
    },
    "NCSC Aug 2020": {
        # outage configurations
        "outage_path":    "ncoutage_202005-09.npy",
        "outage_geo":     "nc_geolocation_115.npy",
        "outage_startt":  "2020-05-01 00:00:00",
        "outage_endt":    "2020-09-14 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_ncscweather-202005",
        "weather_geo":    "nc_geolocation_115.npy",
        "weather_startt": "2020-05-01 00:00:00",
        "weather_endt":   "2020-09-14 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        # Hurricane Isaias
        "_startt":        "2020-07-31 00:00:00",
        "_endt":          "2020-08-10 00:00:00"
    },
}



def load_outage(config, N=4):
    # load geo locations appeared in outage data
    geo_outage = np.load("data/%s" % config["outage_geo"])
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
    geo_weather = np.load("data/%s" % config["weather_geo"])
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



def dataloader(config, standardization=True, outageN=3, weatherN=3, isproj=True):
    """
    data loader for MA data sets including outage sub data set and weather sub data set

    - season: summer or winter
    """
    obs_outage, geo_outage = load_outage(config)
    obs_feats, geo_weather = load_weather(config)

    # # NOTE: FOR NCSC DATA
    # n_locs    = obs_outage.shape[1]
    # obs_feats = obs_feats[:, :, :n_locs]

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
    if isproj:
        obs_feats = [ proj(obs, coord=geo_weather, proj_coord=geo_outage[:, :2], k=10) for obs in obs_feats ]

    obs_outage  = avg(obs_outage, N=outageN).transpose()                    # [ n_locations, n_times ]
    obs_feats   = [ avg(obs, N=weatherN).transpose() for obs in obs_feats ] # ( n_feats, [ n_locations, n_times ] )
    obs_weather = np.stack(obs_feats, 2)                                    # [ n_locations, n_times, n_feats ]

    return obs_outage, obs_weather, geo_outage, geo_weather



if __name__ == "__main__":
    dataloader(config["MA Oct 2019"], standardization=False)