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

from dataloader import MAdataloader

def train(model, niter=1000, lr=1e-1, log_interval=50):
    """training procedure for one epoch"""
    # coordinates of K locations
    coords    = np.load("data/geolocation.npy")[:, :2]
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
        if hasattr(module, 'halpha'):
            halpha = module.halpha.data
            module.halpha.data = torch.clamp(halpha, min=0.)
        if hasattr(module, 'hbeta'):
            hbeta  = module.hbeta.data
            module.hbeta.data  = torch.clamp(hbeta, min=0.)
        if hasattr(module, 'gamma1'):
            gamma1 = module.gamma1.data
            module.gamma1.data = torch.clamp(gamma1, min=0.)
        if hasattr(module, 'gamma2'):
            gamma2 = module.gamma2.data
            module.gamma2.data = torch.clamp(gamma2, min=0.)



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
        return beta * torch.exp(- (t - tp) * beta)



class TorchHawkesCovariates(TorchHawkes):
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
        # self.delta = torch.nn.Parameter(torch.Tensor(torch.ones(1)))
        self.gamma1 = torch.nn.Parameter(torch.Tensor(torch.ones(1)))
        self.gamma2 = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, .01))
        # network
        self.nn     = torch.nn.Sequential(
            torch.nn.Linear(self.M * self.d * 2, 20), # [ M * d * 2, 20 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(20, 1),                   # [ 20, 1 ]
            torch.nn.Softplus())

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
        if _t < self.d or _t + self.d > self.N:
            lam2 = torch.zeros(self.K)                              # [ K ]
        else:
            # covariates
            X    = self.covs[:, _t - self.d:_t + self.d, :].clone() # [ K, d * 2, M ]
            X    = X.reshape(self.K, self.M * self.d * 2)           # [ K, M * d * 2 ]
            lam2 = self.nn(X)                                       # [ K, 1 ]
        lam = self.gamma1 * lam1 + self.gamma2 * lam2.clone().squeeze_()
        return lam

        # # self-exciting effects
        # lam1 = TorchHawkes._lambda(self, _t)
        # # covariate effects
        # lam2 = 0
        # d    = self.d if _t >= self.d else _t
        # # current time and the past 
        # t    = torch.ones((self.K, d, self.M), dtype=torch.int32) * _t               # [ K, d, M ]
        # tp   = torch.arange(d).unsqueeze_(0).unsqueeze_(2).repeat(self.K, 1, self.M) # [ K, d, M ]
        # tp   = _t - tp                                                               # [ K, d, M ]
        # # coefficients for each covariate
        # coef = self._exp_kernel(self.gamma, t, tp) # [ d ]

        # # # covariates
        # # X    = self.covs[:, _t - d:_t, :]          # [ K, d, M ]
        # # X    = X * coef
        # # # intensity
        # # lam2 = self.nn(X.reshape(self.K * d, self.M)).reshape(self.K, d) # [ K, d ]
        # # lam2 = lam2.sum(1)                                               # [ K ]                 
        # # lam  = lam1 + lam2
        # # return lam

        # # covariates
        # X    = self.covs[:, _t - d:_t, :]           # [ K, d, M ]
        # # intensity
        # lam2 = (X * coef).sum(2).sum(1) * self.zeta # [ K ]              
        # lam  = torch.nn.functional.softplus(self.delta * lam1 + lam2)
        # return lam

        


if __name__ == "__main__":
    torch.manual_seed(1)

    import matplotlib.pyplot as plt 
    from plot import plot_data_on_map, plot_2data_on_linechart, plot_beta_net_on_map, error_heatmap

    locs    = np.load("data/geolocation.npy")
    loc_ids = locs[:, 2]
    # obs_outage, obs_temp, obs_wind = MAdataloader(is_training=False)
    start_date, obs_outage, obs_weather = MAdataloader(is_training=True)
    print(obs_outage.shape)
    print(obs_weather.shape)

    model = TorchHawkesCovariates(d=3, obs=obs_outage, covariates=obs_weather)

    train(model, niter=1000, lr=1., log_interval=10)
    print("[%s] saving model..." % arrow.now())
    torch.save(model.state_dict(), "saved_models/hawkes_covariates_future.pt")
    
    print(model.hbeta.detach().numpy())
    print(model.gamma1.detach().numpy())
    print(model.gamma2.detach().numpy())

    _, lams = model()
    lams    = lams.detach().numpy()

    # ---------------------------------------------------
    boston_ind = np.where(loc_ids == 199.)[0][0]
    worces_ind = np.where(loc_ids == 316.)[0][0]
    spring_ind = np.where(loc_ids == 132.)[0][0]
    cambri_ind = np.where(loc_ids == 192.)[0][0]
    plot_2data_on_linechart(start_date, lams.sum(0), obs_outage.sum(0), "Total number of outages over time (training data) with future covariates", dayinterval=1)
    plot_2data_on_linechart(start_date, lams[boston_ind], obs_outage[boston_ind], "Prediction results for Boston, MA (training data) with future covariates", dayinterval=1)
    plot_2data_on_linechart(start_date, lams[worces_ind], obs_outage[worces_ind], "Prediction results for Worcester, MA (training data) with future covariates", dayinterval=1)
    plot_2data_on_linechart(start_date, lams[spring_ind], obs_outage[spring_ind], "Prediction results for Springfield, MA (training data) with future covariates", dayinterval=1)
    plot_2data_on_linechart(start_date, lams[cambri_ind], obs_outage[cambri_ind], "Prediction results for Cambridge, MA (training data) with future covariates", dayinterval=1)
    # ---------------------------------------------------



    # # ---------------------------------------------------
    # locs_order = np.argsort(loc_ids)
    # error_heatmap(real_data=obs_outage, pred_data=lams, locs_order=locs_order, start_date=start_date, dayinterval=1, modelname="ST-Cov-train")
    # # ---------------------------------------------------
    