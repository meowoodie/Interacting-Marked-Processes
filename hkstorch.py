#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import random
import torchutils
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from scipy import sparse
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances

from dataloader import dataloader, config

def train(model, locs, niter=1000, lr=1e-1, log_interval=50):
    """training procedure for one epoch"""
    # coordinates of K locations
    coords    = locs[:, :2]
    # define model clipper to enforce inequality constraints
    clipper1  = NonNegativeClipper()
    clipper2  = ProximityClipper(coords, k=100)
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    logliks = []
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for _iter in range(niter):
        try:
            model.train()
            optimizer.zero_grad()           # init optimizer (set gradient to be zero)
            loglik, _, _ = model()
            # objective function
            loss         = - loglik
            loss.backward()                 # gradient descent
            optimizer.step()                # update optimizer
            model.apply(clipper1)
            model.apply(clipper2)
            # log training output
            logliks.append(loglik.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train batch: %d\tLoglik: %.3e" % (arrow.now(), 
                    _iter / log_interval, 
                    sum(logliks) / log_interval))
                logliks = []
        except KeyboardInterrupt:
            break



class NonNegativeClipper(object):
    """
    References:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/3
    """

    def __init__(self):
        pass

    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Alpha'):
            Alpha = module.Alpha.data
            module.Alpha.data = torch.clamp(Alpha, min=0.)
        if hasattr(module, 'Beta'):
            Beta  = module.Beta.data
            module.Beta.data  = torch.clamp(Beta, min=0.)
        # TorchHawkesNNCovariates
        if hasattr(module, 'Gamma'):
            Gamma  = module.Gamma.data
            module.Gamma.data = torch.clamp(Gamma, min=0.)
        if hasattr(module, 'Omega'):
            Omega  = module.Omega.data
            module.Omega.data = torch.clamp(Omega, min=0.)



class ProximityClipper(object):
    """
    """

    def __init__(self, coords, k):
        """
        Args:
        - coords: a list of coordinates for K locations [ K, 2 ]
        """
        distmat      = euclidean_distances(coords)        # [K, K]
        proxmat      = self._k_nearest_mask(distmat, k=k) # [K, K]
        self.proxmat = torch.FloatTensor(proxmat)         # [K, K]
        
    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Alpha'):
            alpha = module.Alpha.data
            module.Alpha.data = alpha * self.proxmat
    
    @staticmethod
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



class TorchHawkes(torch.nn.Module):
    """
    PyTorch Module for Hawkes Processes
    """

    def __init__(self, obs):
        """
        Denote the number of time units as N, the number of locations as K

        Args:
        - obs:    event observations    [ N, K ]
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(obs) # [ K, N ]
        # configurations
        self.K, self.N = obs.shape
        # parameters
        self.Mu0   = self.obs.mean(1) / 10 + 1e-2                                      # [ K ]
        self.Beta  = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3))           # [ K ]
        self.Alpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01)) # [ K, K ]
    
    def _mu(self, _t):
        """
        Background rate at time `t`
        """
        return self.Mu0

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            # current time and the past 
            t      = torch.ones(_t, dtype=torch.int32) * _t      # [ t ]
            tp     = torch.arange(_t)                            # [ t ]
            # self-exciting effect
            kernel = self.__exp_kernel(self.Beta, t, tp, self.K) # [ K, t ]
            Nt     = self.obs[:, :_t].clone()                    # [ K, t ]
            lam    = torch.mm(self.Alpha, Nt * kernel).sum(1)    # [ K ]
            lam    = torch.nn.functional.softplus(lam)           # [ K ]
        else:
            lam    = torch.zeros(self.K)
        return lam
        
    def _log_likelihood(self):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   a list of historical conditional intensity values at time t = tau, ..., t
        """
        # lambda values from 0 to N
        lams0    = [ self._mu(t) for t in np.arange(self.N) ]     # ( N, [ K ] )
        lams1    = [ self._lambda(t) for t in np.arange(self.N) ] # ( N, [ K ] )
        lams0    = torch.stack(lams0, dim=1)                      # [ K, N ]
        lams1    = torch.stack(lams1, dim=1)                      # [ K, N ]
        Nloglams = self.obs * torch.log(lams0 + lams1 + 1e-5)     # [ K, N ]
        # log-likelihood function
        loglik   = (Nloglams - lams0 - lams1).sum()
        return loglik, lams0, lams1

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()

    @staticmethod
    def __exp_kernel(Beta, t, tp, K):
        """
        Args:
        - Beta:  decaying rate [ K ]
        - t, tp: time index    [ t ]
        """
        delta_t = t - tp                              # [ t ]
        delta_t = delta_t.unsqueeze(0).repeat([K, 1]) # [ K, t ]
        Beta    = Beta.unsqueeze(1)                   # [ K, 1 ]
        return Beta * torch.exp(- delta_t * Beta)



class TorchHawkesNNCovariates(TorchHawkes):
    """
    PyTorch Module for Hawkes Processes with Externel Observation
    """

    def __init__(self, d, obs, covariates):
        """
        Denote the number of time units as N, the number of locations as K, and 
        the number of externel features as M.

        Args:
        - d:      memory depth
        - obs:    event observations    [ N, K ]
        - extobs: externel observations [ N, K, M ]
        """
        TorchHawkes.__init__(self, obs)
        # configuration
        self.d       = d                        # d: memory depth
        K, N, self.M = covariates.shape         # M: number of covariates
        assert N == self.N and K == self.K, \
            "invalid dimension (%d, %d, %d) of covariates, where N is not %d or K is not %d." % \
            (N, K, self.M, self.N, self.K)
        # data
        self.covs  = torch.Tensor(covariates)   # [ K, N, M ]
        # parameters
        self.Gamma = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, .01)) # [ K ]
        self.Omega = torch.nn.Parameter(torch.Tensor(self.M).uniform_(0, .5))  # [ M ]
        # network
        self.nn    = torch.nn.Sequential(
            torch.nn.Linear(self.M, 200),       # [ M, 20 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(200, 200),          # [ 20, 1 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(200, 1),            # [ 20, 1 ]
            torch.nn.Softplus())
        self.hmu   = 0

    def _mu(self, _t):
        """
        Background rate at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        # get covariates in the past d time slots
        if _t < self.d:
            X     = self.covs[:, :_t, :].clone()                            # [ K, t, M ]
            X_pad = self.covs[:, :1, :].clone().repeat([1, self.d - _t, 1]) # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                            # [ K, d, M ]
        else:
            X     = self.covs[:, _t-self.d:_t, :].clone()                   # [ K, d, M ]
        # convolution with an exponential decaying kernel
        conv_X    = self.conv_exp_decay_kernel(X)                           # [ K, M ]
        # calculate base intensity
        mu  = self.nn(conv_X)                                               # [ K, 1 ]
        mu  = self.Gamma * mu.clone().squeeze_()                            # [ K ]
        return mu

    def conv_exp_decay_kernel(self, X):
        """
        Compute convolution of covariates with an exponential decaying kernel.

        Arg:
        - X: observed covariates in the past d time slots [ K, d, M ]
        """
        # exponential decaying kernel
        delta_t = torch.arange(self.d)                       # [ d ]
        delta_t = delta_t.unsqueeze(1).repeat([1, self.M])   # [ d, M ]
        Omega   = self.Omega.unsqueeze(0)                    # [ 1, M ]
        kernel  = torch.exp(- delta_t * Omega)               # [ d, M ]
        kernel  = kernel.unsqueeze(0).repeat([self.K, 1, 1]) # [ K, d, M ]
        # convolution 
        conv_X  = (X * kernel).sum(1)                        # [ K, M ]
        return conv_X



if __name__ == "__main__":
    
    torch.manual_seed(2)
    
    from plot_ma import *

    # load data
    obs_outage, obs_weather, locs, _ = dataloader(config["MA Mar 2018"])
    loc_ids = locs[:, 2]

    # training
    model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    model.load_state_dict(torch.load("saved_models/hawkes_covariates_vecbeta_ma_201803full_hisd24_feat35.pt"))
    # train(model, locs=locs, niter=1000, lr=1., log_interval=10)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/hawkes_covariates_vecbeta_ma_201803full_hisd24_feat35.pt")

    # evaluation
    _, mus, lams = model()
    # lams         = lams.detach().numpy()
    # mus          = mus.detach().numpy()
    # lams         = lams + mus

    # # visualization
    # boston_ind = np.where(loc_ids == 199.)[0][0]
    # worces_ind = np.where(loc_ids == 316.)[0][0]
    # spring_ind = np.where(loc_ids == 132.)[0][0]
    # cambri_ind = np.where(loc_ids == 192.)[0][0]
    # plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], obs_outage.sum(0), lams.sum(0), "Prediction of total outages in MA (Mar 2018)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], obs_outage[boston_ind], lams[boston_ind], "Prediction for Boston, MA (Mar 2018)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], obs_outage[worces_ind], lams[worces_ind], "Prediction for Worcester, MA (Mar 2018)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], obs_outage[spring_ind], lams[spring_ind], "Prediction for Springfield, MA (Mar 2018)", dayinterval=1)
    # plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], obs_outage[cambri_ind], lams[cambri_ind], "Prediction for Cambridge, MA (Mar 2018)", dayinterval=1)
    # print(model.Omega)
    # print(model.Omega.mean())

    

    obs_outage, obs_weather, locs, _ = dataloader(config["Normal MA Mar 2018"], standardization=False)
    ncust        = np.expand_dims(np.load("data/ncustomer_ma.npy"), axis=1)
    conv_weather = torchutils.conv_covariates(obs_weather, model)

    gamma  = model.Gamma.detach().numpy()
    mask   = np.where(gamma > 0)[0]
    thoriz = list(range(0,18)) + list(range(120,150))

    # find the cities with the largest and the smallest disruption rates
    rates  = obs_outage[mask, :].max(axis=1) / ncust[mask, 0]

    for i in range(35):
        _id = (- 1 * rates).argsort()

        x   = conv_weather[mask, :, :].detach().numpy()
        X   = x[:, thoriz, i] # .reshape(len(mask) * len(thoriz))

        y   = obs_outage[mask, :] / ncust[mask, :]
        Y   = y[:, thoriz] # .reshape(len(mask) * len(thoriz))

        # fig = plt.figure(figsize=(8, 8))
        # plt.scatter(X, Y, s=2, c="gray", alpha=0.3)

        fig  = plt.figure()
        ax   = fig.add_subplot(111, projection='3d')
        cmap = matplotlib.cm.get_cmap('Reds')
        for z, j in enumerate(_id):
            _X = X[j, :]
            _Y = Y[j, :]
            _Z = np.ones(len(thoriz)) * z
            ax.scatter(_Z, _X, _Y, c=_Y, cmap=cmap, vmin=0, vmax=1)
            ax.set_zlim(0, 1)
        plt.show()
        # plt.savefig("%d-weather.png" % i)