#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot import *

from dataloader import load_outage, load_weather, dataloader, config
from hkstorch import TorchHawkes, TorchHawkesNNCovariates

if __name__ == "__main__":

    # obs_outage, obs_weather, locs = dataloader(config["MA Mar 2018"])
    # loc_ids = locs[:, 2]

    # # model1 = TorchHawkes(obs=obs_outage)
    # model2 = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)

    # # model1.load_state_dict(torch.load("saved_models/hawkes.pt"))
    # model2.load_state_dict(torch.load("saved_models/hawkes_covariates_varbeta_ma_201803full_d6_feat35.pt"))

    # # _, lams1 = model1()
    # # lams1    = lams1.detach().numpy()

    # _, lams2 = model2()
    # lams2    = lams2.detach().numpy()

    # ---------------------------------------------------
    # #  Plot data

    # plot_illustration(locs)
    # plot_data_exp_decay(locs, obs_outage)
    # plot_data_constant_alpha(locs, obs_outage, loc_ids)
    # ---------------------------------------------------
    


    # # ---------------------------------------------------
    # #  Plot temporal predictions

    # boston_ind = np.where(loc_ids == 199.)[0][0]
    # worces_ind = np.where(loc_ids == 316.)[0][0]
    # spring_ind = np.where(loc_ids == 132.)[0][0]
    # cambri_ind = np.where(loc_ids == 192.)[0][0]
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2.sum(0), obs_outage.sum(0), "Prediction of total outages in MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[boston_ind], obs_outage[boston_ind], "Prediction for Boston, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[worces_ind], obs_outage[worces_ind], "Prediction for Worcester, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[spring_ind], obs_outage[spring_ind], "Prediction for Springfield, MA (Oct 2019)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Oct 2019"]["_startt"], lams2[cambri_ind], obs_outage[cambri_ind], "Prediction for Cambridge, MA (Oct 2019)", dayinterval=1)
    # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot error matrix

    # locs_order = np.argsort(loc_ids)
    # error_heatmap(real_data=obs_outage, pred_data=lams2, locs_order=locs_order, start_date=config["MA Mar 2018"]["_startt"], dayinterval=1, modelname="our model feat 43")
    # error_heatmap(real_data=obs_outage, pred_data=lams1, locs_order=locs_order, start_date=start_date, dayinterval=1, modelname="Hawkes without feat")
    # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot gamma

    # gamma = model2.gamma.detach().numpy()
    # plot_data_on_map_in_color(gamma, locs, "Weather coefficients")
    # # ---------------------------------------------------



    # # # ---------------------------------------------------
    # # #  Plot Alpha

    # alpha = model2.halpha.detach().numpy()
    # save_significant_alpha(model2, loc_ids, obs_outage)
    # # plot_data_on_map_in_color(alpha.sum(0), locs, "Critical cities")
    # # plot_data_on_map_in_color(alpha.sum(1), locs, "Vulnerable cities")
    # # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot param space

    # _, obs_weather, _ = dataloader(config["MA Mar 2018"], standardization=False)
    # plot_nn_params(model2, obs_weather)
    # plot_nn_3Dparams(model2, obs_weather)
    # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot base intensity

    # plot_baselines_and_lambdas(model2, config["MA Mar 2018"]["_startt"], obs_outage)
    # plot_spatial_base(model2, locs, obs_outage)
    # plot_spatial_lam_minus_base(model2, locs, obs_outage)
    # plot_spatial_ratio(model2, locs, obs_outage)
    # plot_spatial_base_and_cascade(model2, locs, obs_outage)
    # plot_spatial_base_and_cascade_over_time(model2, locs, obs_outage)
    # # ---------------------------------------------------

    # # ---------------------------------------------------
    # # Plot outage and weather on a line chart

    N      = 129
    feats  = [6, -4]
    colors = ["#DC143C", "#0165fc"] #, "#3f9b0b"]
    labels = ["Derived radar reflectivity", "Wind speed"]
    obs_outage, obs_weather, _ = dataloader(config["Normal MA Mar 2018"], standardization=False, weatherN=1) 
    obs_outage                 = obs_outage[:, :N]
    obs_weather_show           = obs_weather[:, :N*3, feats]
    obs_weather_normal         = obs_weather[:, N*3:220*3, feats]
    plot_outage_and_weather_linechart(
        N, config["MA Mar 2018"]["_startt"], 
        obs_outage, obs_weather_show, obs_weather_normal, labels, colors, 
        dayinterval=3)

    # N      = 129
    # feat   = 6
    # obs_outage, geo_outage = load_outage(config["Normal MA Mar 2018"], N=1)
    # obs_feats, geo_weather = load_weather(config["Normal MA Mar 2018"])
    # obs_outage             = obs_outage[:N, :].sum(0)
    # obs_weather_show       = obs_feats[feat, :N, :].mean(0)
    # obs_weather_normal     = obs_feats[feat, N*3:220*3, :].mean(0)
    # # obs_weather_show       = obs_feats[feat, 25, :]
    # # obs_weather_normal     = obs_feats[feat, 0, :]
    # plot_outage_and_weather_map(geo_outage, geo_weather, obs_outage, obs_weather_show, obs_weather_normal)

    # # ---------------------------------------------------
