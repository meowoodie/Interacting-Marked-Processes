#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from dataloader import load_outage, load_weather, dataloader, config
from hkstorch import TorchHawkesNNCovariates, train
from utils import avg



def reweight_omega(model_copy, factor):
    """
    reweight omega for weather variables. 
    """
    old_omega = model_copy.Omega.detach().numpy()
    for _id in range(model.M):
        model_copy.Omega[_id] = factor # old_omega[_id] * factor
    return model_copy



def reweight_beta(model_copy, n=5):
    """
    reweight top `n` units with largest gamma values to the average of beta. 
    """
    # the average of gamma
    avg_beta = model_copy.Beta.mean()
    # sort gamma
    srt_ids   = model_copy.Beta.argsort()
    # reweight top n to the average
    for _id in srt_ids[:n]:
        if model_copy.Beta[_id] < avg_beta:
            model_copy.Beta[_id] = avg_beta
    return model_copy



def reweight_gamma(model_copy, n=5):
    """
    reweight top `n` units with largest gamma values to the average of gamma. 
    """
    # the average of gamma
    avg_gamma = model_copy.Gamma.mean()
    # sort gamma
    srt_ids   = model_copy.Gamma.argsort()
    # reweight top n to the average
    for _id in srt_ids[model_copy.K-n:]:
        if model_copy.Gamma[_id] > avg_gamma:
            model_copy.Gamma[_id] = avg_gamma
    return model_copy



def reweight_alpha(model_copy, k=5, n=5):
    """
    reweight top `n` units with largest alpha values to the average of alpha. 
    """
    # the average of gamma
    avg_alpha = model_copy.Alpha.mean()
    med_alpha = model_copy.Alpha.median()
    # top k units by their maximum number of outages
    src_ids   = model_copy.obs.max(1)[0].argsort()[model_copy.K-k:]
    for s_id in src_ids:
        # top n units by their alpha values given the source `s_id`
        trg_ids = model_copy.Alpha[s_id].argsort()[model_copy.K-n:]
        for t_id in trg_ids:
            if model_copy.Alpha[s_id, t_id] >= avg_alpha:
                model_copy.Alpha[s_id, t_id] = avg_alpha
    return model_copy



def simulation(model):
    # evaluate model
    _, mus, triggs = model()
    # compute recovered outages
    triggs = triggs.detach().numpy().sum(0)
    mus    = mus.detach().numpy().sum(0)
    lams   = triggs + mus
    return lams.max()



def plot_matrix_sim_results(mat, K, N):
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        'weight' : 'bold',
        'size'   : 25}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    mat = 1 - mat / mat.max()

    fig  = plt.figure(figsize=(10, 10))
    ax   = fig.add_subplot(111, projection='3d')

    X = np.arange(0, K, 1)
    Y = np.arange(0, N, 1)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, mat, cmap=cm.winter_r, alpha=0.75)

    ax.set_xlim(0, K)
    ax.set_ylim(N, 0)
    ax.set_zlim(0.5, 0)

    ax.set_zlabel("Percentage of outage reduction", labelpad=35)
    ax.set_xlabel("Top units by\nnumber of outages", labelpad=30)
    ax.set_ylabel("Reweighted edges\nper unit", labelpad=30)
    ax.tick_params(axis='z', which='major', pad=20)
    ax.tick_params(axis='x', which='major', pad=10)
    ax.tick_params(axis='y', which='major', pad=10)
    # # ax.grid(False)

    ax.set_xticks(np.arange(0, K+1, 25))
    ax.set_yticks(np.arange(0, N+1, 25))
    ax.set_zticks([0., 0.25, 0.5])
    ax.set_xticklabels(["%d" % d for d in np.arange(0, K+1, 25)])
    ax.set_yticklabels(["%d" % d for d in np.arange(0, N+1, 25)])
    ax.set_zticklabels(["0\%", "25\%", "50\%"])

    # plt.title("GA in October 2018", pad=20)
    plt.title("NC \& SC in August 2020", pad=20)
    # plt.title("MA in March 2018", pad=20)
    fig.tight_layout()
    # plt.savefig("imgs/ga_simulation_reweight_alpha.pdf")
    plt.savefig("imgs/ncsc_simulation_reweight_alpha.pdf")
    # plt.savefig("imgs/ma_simulation_reweight_alpha.pdf")



def plot_vector_sim_results(vec_gamma, vec_beta, vec_omega, max_omega, N, titlename, filename):
    font = {
        'family' : 'sans-serif',
        'sans-serif': ['Helvetica'],
        'weight' : 'bold',
        'size'   : 25}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)

    fig = plt.figure(figsize=(10, 10))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    gs  = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    
    vec_gamma = 1 - vec_gamma / vec_gamma.max()
    vec_beta  = 1 - vec_beta / vec_beta.max()
    vec_omega = 1 - vec_omega / max_omega
    print(vec_omega)

    ax1.fill_between(np.arange(0, N, 1), vec_gamma, np.zeros(N), where=vec_gamma > np.zeros(N), facecolor='red', alpha=0.2, interpolate=True)
    ax1.plot(np.arange(0, N, 1), vec_gamma, linewidth=5, color="red", alpha=1, linestyle="--")

    ax2.fill_between(np.arange(0, N, 1), vec_beta, np.zeros(N), where=vec_beta > np.zeros(N), facecolor='green', alpha=0.2, interpolate=True)
    ax2.plot(np.arange(0, N, 1), vec_beta, linewidth=5, color="green", alpha=1, linestyle="--")

    ax3.fill_between(np.arange(0, N, 1), vec_omega, np.zeros(N), where=vec_omega > np.zeros(N), facecolor='blue', alpha=0.2, interpolate=True)
    ax3.fill_between(np.arange(0, N, 1), vec_omega, np.zeros(N), where=vec_omega < np.zeros(N), facecolor='gray', alpha=0.2, interpolate=True)
    ax3.plot(np.arange(0, N, 1), np.zeros(N), linewidth=4, color="black", alpha=1, linestyle=":")
    ax3.plot(np.arange(0, N, 1), vec_omega, linewidth=5, color="blue", alpha=1, linestyle="--")

    ax1.set_xlabel("Top units\nby vulnerability")
    ax2.set_xlabel("Worst units\nby recoverability")
    ax3.set_xlabel("Average decay rate $\omega$ in weather effect")
    ax1.set_ylabel("Percentage of\noutage reduction")
    ax3.set_ylabel("Percentage of\noutage reduction")
    ax1.set_xticks(np.arange(0, N+1, 25))
    ax1.set_xticklabels(["%d" % d for d in np.arange(0, N+1, 25)])
    ax2.set_xticks(np.arange(0, N+1, 25))
    ax2.set_xticklabels(["%d" % d for d in np.arange(0, N+1, 25)])
    ax3.set_xticks([0, 25, 50])
    ax3.set_xticklabels(["0.0", "0.5", "1.0"])

    ax1.set_ylim(0.52, -0.02)
    ax1.set_yticks([0., 0.25, 0.5])
    ax1.set_yticklabels(["0\%", "25\%", "50\%"])
    ax2.set_ylim(0.0251, -0.001)
    ax2.set_yticks([0., 0.0125, 0.025])
    ax2.set_yticklabels(["0\%", "1.25\%", "2.50\%"])
    # # ncsc
    # ax3.set_ylim(1.2, -7.2)
    # ax3.set_yticks([-7, -3.5, 0., 1.])
    # ax3.set_yticklabels(["-700\%", "-350\%", "0\%", "100\%"])
    # ga
    ax3.set_ylim(0.23, -1.03)
    ax3.set_yticks([-1., 0., .2])
    ax3.set_yticklabels(["-100\%", "0\%", "20\%"])
    # # ma
    # ax3.set_ylim(1.125, -4.125)
    # ax3.set_yticks([-4., -2., 0., 1.])
    # ax3.set_yticklabels(["-400\%", "-200\%", "0\%", "100\%"])

    fig.suptitle(titlename)
    fig.tight_layout()

    fig.subplots_adjust(top=.88)
    plt.savefig("imgs/vert_%s.pdf" % filename)




if __name__ == "__main__":

    # obs_outage, obs_weather, _, _ = dataloader(
    #     config["GA Oct 2018"], standardization=True, outageN=3, weatherN=3, isproj=False)
    # print(obs_outage.sum(0).max())
    # model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    # model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ga_201810_hisd24_feat35.pt"))

    # obs_outage, obs_weather, _, _ = dataloader(config["NCSC Aug 2020"], outageN=3, weatherN=3, isproj=False)
    # print(obs_outage.sum(0).max())
    # model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    # model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ncsc_202008_hisd24_feat35.pt"))

    # obs_outage, obs_weather, _, _ = dataloader(config["MA Mar 2018"])
    # print(obs_outage.sum(0).max())
    # model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    # model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ma_201803full_hisd24_feat35.pt"))


    # # SIMULATION FOR REWEIGHT OMEGA
    # N   = 50
    
    # vec = np.zeros(N)
    # with torch.no_grad():
    #     print(model.Omega.mean())
    #     max_val = simulation(model)
    #     for i, n in enumerate(np.linspace(0, 1, N)):
    #         model_copy = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    #         model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
    #         model_copy = reweight_omega(model_copy, factor=n)
    #         val    = simulation(model_copy)
    #         vec[i] = val
    #         print("[%s] n=%f, maximum outage=%f" % (arrow.now(), n, val))
    #     np.save("data/ma_omega_simulation.npy", vec)
    #     plt.plot(1 - vec / max_val, c="blue")
    #     plt.savefig("imgs/ma_omega_test.pdf")



    # SIMULATION FOR REWEIGHT BETA

    # N = 50

    # vec = np.zeros(N)
    # with torch.no_grad():
    #     max_val = simulation(model)
    #     print(max_val)
    #     for n in np.arange(0, N, 1):
    #         model_copy = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    #         model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
    #         model_copy = reweight_beta(model_copy, n=n)
    #         val = simulation(model_copy)
    #         vec[n] = val if val < max_val else max_val
    #         print("[%s] n=%d, maximum outage=%f" % (arrow.now(), n, vec[n]))
    #     np.save("data/ma_beta_simulation.npy", vec)
    #     plt.plot(vec)
    #     plt.savefig("imgs/ma_beta_test.pdf")



    # SIMULATION FOR REWEIGHT GAMMA

    # N  = 50

    # vec = np.zeros(N)
    # with torch.no_grad():
    #     for n in np.arange(0, N, 1):
    #         model_copy = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    #         model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
    #         model_copy = reweight_gamma(model_copy, n=n)
    #         val = simulation(model_copy)
    #         vec[n] = val
    #         print("[%s] n=%d, maximum outage=%f" % (arrow.now(), n, val))
    #     np.save("data/ma_gamma_simulation.npy", vec)
    #     plt.plot(vec)
    #     plt.savefig("imgs/ma_gamma_test.pdf")



    # SIMULATION FOR REWEIGHT ALPHA

    # K, N = 50, 50

    # img  = np.zeros((K, N))
    # with torch.no_grad():
    #     for k in np.arange(0, K, 1):
    #         for n in np.arange(0, N, 1):
    #             print("[%s] k=%d, n=%d" % (arrow.now(), k, n))
    #             model_copy = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    #             model_copy.load_state_dict(model.state_dict()) # copy weights and stuff
    #             model_copy = reweight_alpha(model_copy, k=k, n=n)
    #             val = simulation(model_copy)
    #             img[k, n] = val
    #             print("[%s] maximum outage: %f" % (arrow.now(), val))
    #     np.save("data/ma_alpha_simulation.npy", img)
    #     plt.imshow(img)
    #     plt.savefig("imgs/ma_test.pdf")



    # VISUALIZATION

    N = 50

    ga_beta_vec = np.load("data/ga_beta_simulation.npy")
    nc_beta_vec = np.load("data/ncsc_beta_simulation.npy")
    ma_beta_vec = np.load("data/ma_beta_simulation.npy")

    ga_gamma_vec = np.load("data/ga_gamma_simulation.npy")
    nc_gamma_vec = np.load("data/ncsc_gamma_simulation.npy")
    ma_gamma_vec = np.load("data/ma_gamma_simulation.npy")

    ga_omega_vec = np.load("data/ga_omega_simulation.npy")
    nc_omega_vec = np.load("data/ncsc_omega_simulation.npy")
    ma_omega_vec = np.load("data/ma_omega_simulation.npy")

    # avg omega 0.2807
    plot_vector_sim_results(ga_gamma_vec, ga_beta_vec, ga_omega_vec, 193018, N, titlename="GA in October 2020", filename="ga_simulation_reweight")
    # avg omega 0.2541
    # plot_vector_sim_results(nc_gamma_vec, nc_beta_vec, nc_omega_vec, 202217, N, titlename="NC \& SC in August 2020", filename="ncsc_simulation_reweight")
    # avg omega 0.2956
    # plot_vector_sim_results(ma_gamma_vec, ma_beta_vec, ma_omega_vec, 420723, N, titlename="MA in March 2020", filename="ma_simulation_reweight")

    