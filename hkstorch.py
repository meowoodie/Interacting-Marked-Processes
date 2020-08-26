#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from scipy import sparse
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
            loglik, _ = model()
            # objective function
            loss      = - loglik
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
        if hasattr(module, 'halpha'):
            halpha = module.halpha.data
            module.halpha.data = torch.clamp(halpha, min=0.)
        if hasattr(module, 'hbeta'):
            hbeta  = module.hbeta.data
            module.hbeta.data  = torch.clamp(hbeta, min=0.)
        # TorchHawkesNNCovariates
        if hasattr(module, 'gamma'):
            gamma  = module.gamma.data
            module.gamma.data = torch.clamp(gamma, min=0.)
        if hasattr(module, 'omega'):
            omega  = module.omega.data
            module.omega.data = torch.clamp(omega, min=0.)




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
        if hasattr(module, 'halpha'):
            alpha = module.halpha.data
            module.halpha.data = alpha * self.proxmat
    
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
        self.hmu    = self.obs.mean(1) / 10 + 1e-2
        # self.hbeta  = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, 1))
        self.hbeta  = torch.nn.Parameter(torch.Tensor([2]))
        self.halpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01))

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        # current time and the past 
        t      = torch.ones(_t, dtype=torch.int32) * _t    # [ t ]
        tp     = torch.arange(_t)                          # [ t ]
        # self-exciting effect
        kernel = self._exp_kernel(self.hbeta, t, tp)       # [ t ]
        Nt     = self.obs[:, :_t].clone()                  # [ K, t ]
        lam1   = torch.mm(self.halpha, Nt * kernel).sum(1) # [ K ]
        lam    = torch.nn.functional.softplus(self.hmu + lam1)
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
        lams     = [ self._lambda(t) for t in np.arange(self.N) ] # ( N, [ K ] )
        lams     = torch.stack(lams, dim=1)                       # [ K, N ]
        Nloglams = self.obs * torch.log(lams)                     # [ K, N ]
        # log-likelihood function
        loglik   = (Nloglams - lams).sum()
        return loglik, lams

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()

    @staticmethod
    def _exp_kernel(beta, t, tp):
        """
        Args:
        - t, tp: time index [ n ]
        """
        # print(t - tp)
        # print(beta)
        # print((t - tp) * beta)
        # print(torch.exp(- (t - tp) * beta))
        return beta * torch.exp(- (t - tp) * beta)



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
        # parameter
        self.gamma = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, .01))
        # network
        self.nn    = torch.nn.Sequential(
            torch.nn.Linear(self.M * self.d * 2, 20), # [ M * d * 2, 20 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(20, 1),                   # [ 20, 1 ]
            torch.nn.Softplus())
        self.hmu   = 0

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        # self-exciting effects
        lam1 = TorchHawkes._lambda(self, _t)
        # covariate effects
        if _t < self.d:
            X     = self.covs[:, :_t + self.d, :].clone()                              # [ K, t + d, M ]
            X_pad = self.covs[:, :1, :].clone().repeat([1, self.d - _t, 1])            # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                                       # [ K, d * 2, M ]
        elif _t > self.N - self.d:
            X     = self.covs[:, _t - self.d:, :].clone()                              # [ K, d + N - t, M ]
            X_pad = self.covs[:, -1:, :].clone().repeat([1, self.d + _t - self.N , 1]) # [ K, d + t - N, M ]
            X     = torch.cat([X, X_pad], dim=1)                                       # [ K, d * 2, M ]
        else:
            X  = self.covs[:, _t - self.d:_t + self.d, :].clone() # [ K, d * 2, M ]
        # calculate base intensity
        X   = X.reshape(self.K, self.M * self.d * 2)           # [ K, M * d * 2 ]
        mu  = self.nn(X)                                       # [ K, 1 ]
        # calculate intensity
        lam = lam1 + self.gamma * mu.clone().squeeze_()
        return lam



if __name__ == "__main__":
    
    torch.manual_seed(1)
    
    from plot import *

    # load data
    obs_outage, obs_weather, locs = dataloader(config["Normal MA Mar 2018"])
    loc_ids = locs[:, 2]
    print(obs_outage.shape)
    print(obs_weather.shape)

    # training
    model = TorchHawkesNNCovariates(d=6, obs=obs_outage, covariates=obs_weather)
    train(model, locs=locs, niter=1000, lr=1., log_interval=10)
    print("[%s] saving model..." % arrow.now())
    torch.save(model.state_dict(), "saved_models/hawkes_covariates_varbeta_ma_201803full_d6_feat35.pt")
    print(model.hbeta.detach().numpy())

    # evaluation
    _, lams = model()
    lams    = lams.detach().numpy()

    # visualization
    boston_ind = np.where(loc_ids == 199.)[0][0]
    worces_ind = np.where(loc_ids == 316.)[0][0]
    spring_ind = np.where(loc_ids == 132.)[0][0]
    cambri_ind = np.where(loc_ids == 192.)[0][0]
    plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], lams.sum(0), obs_outage.sum(0), "Prediction of total outages in MA (Mar 2018)", dayinterval=1)
    plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], lams[boston_ind], obs_outage[boston_ind], "Prediction for Boston, MA (Mar 2018)", dayinterval=1)
    plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], lams[worces_ind], obs_outage[worces_ind], "Prediction for Worcester, MA (Mar 2018)", dayinterval=1)
    plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], lams[spring_ind], obs_outage[spring_ind], "Prediction for Springfield, MA (Mar 2018)", dayinterval=1)
    plot_2data_on_linechart(config["MA Mar 2018"]["_startt"], lams[cambri_ind], obs_outage[cambri_ind], "Prediction for Cambridge, MA (Mar 2018)", dayinterval=1)
    