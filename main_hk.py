#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt 
from plot import *

from dataloader import dataloader, config
from hkstorch import TorchHawkes, TorchHawkesNNCovariates

if __name__ == "__main__":

    obs_outage, obs_weather, locs = dataloader(config["MA Mar 2018"])
    loc_ids = locs[:, 2]

    # model1 = TorchHawkes(obs=obs_outage)
    model2 = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)

    # model1.load_state_dict(torch.load("saved_models/hawkes.pt"))
    model2.load_state_dict(torch.load("saved_models/hawkes_covariates_ma_201803_d6.pt"))

    # _, lams1 = model1()
    # lams1    = lams1.detach().numpy()

    _, lams2 = model2()
    lams2    = lams2.detach().numpy()

    # ---------------------------------------------------
    # #  Plot data

    # plot_illustration(locs)
    # plot_data_exp_decay(obs_outage)
    # plot_data_constant_alpha(obs_outage, loc_ids)
    # ---------------------------------------------------
    


    # # ---------------------------------------------------
    # #  Plot temporal predictions

    # boston_ind = np.where(loc_ids == 199.)[0][0]
    # worces_ind = np.where(loc_ids == 316.)[0][0]
    # spring_ind = np.where(loc_ids == 132.)[0][0]
    # cambri_ind = np.where(loc_ids == 192.)[0][0]
    # print(boston_ind, worces_ind, spring_ind, cambri_ind)
    # plot_2data_on_linechart(start_date, lams.sum(0), obs_outage.sum(0), "Total number of outages over time (testing data)", dayinterval=7)
    # plot_2data_on_linechart(start_date, lams[boston_ind], obs_outage[boston_ind], "Prediction results for Boston, MA (testing data)", dayinterval=7)
    # plot_2data_on_linechart(start_date, lams[worces_ind], obs_outage[worces_ind], "Prediction results for Worcester, MA (testing data)", dayinterval=7)
    # plot_2data_on_linechart(start_date, lams[spring_ind], obs_outage[spring_ind], "Prediction results for Springfield, MA (testing data)", dayinterval=7)
    # plot_2data_on_linechart(start_date, lams[cambri_ind], obs_outage[cambri_ind], "Prediction results for Cambridge, MA (testing data)", dayinterval=7)
    # # ---------------------------------------------------



    # # ---------------------------------------------------
    # #  Plot error matrix

    # locs_order = np.argsort(loc_ids)
    # # error_heatmap(real_data=obs_outage, pred_data=lams2, locs_order=locs_order, start_date=start_date, dayinterval=1, modelname="ST-Cov-NN-future-upt (d=6)")
    # error_heatmap(real_data=obs_outage, pred_data=lams1, locs_order=locs_order, start_date=start_date, dayinterval=1, modelname="error-Hawkes")
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

    # _, _, obs_weather = MAdataloader(is_training=True, standardization=False)
    # plot_nn_params(model2, obs_weather)
    # # ---------------------------------------------------



    # ---------------------------------------------------
    #  Plot base intensity

    plot_baselines_and_lambdas(model2, obs_outage)
    plot_spatial_base(model2, locs, obs_outage)
    plot_spatial_lam_minus_base(model2, locs, obs_outage)
    plot_spatial_ratio(model2, locs, obs_outage)
    # ---------------------------------------------------