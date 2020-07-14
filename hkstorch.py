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



class TorchHawkes(torch.nn.Module):
    """
    PyTorch Module for Hawkes Processes with Externel Observation
    """

    def __init__(self, d, obs, extobs=None):
        """
        Denote the number of time units as N, the number of locations as K, and 
        the number of externel features as M.

        Args:
        - d:      memory depth
        - obs:    event observations    [ N, K ]
        - extobs: externel observations [ N, K, M ]
        """
        torch.nn.Module.__init__(self)
        # configurations
        # self.N, self.K, self.M = extobs.shape
        self.K, self.N = obs.shape
        self.d         = d
        # data
        self.obs    = torch.Tensor(obs) # [ K, N ]
        self.extobs = torch.Tensor(extobs) if extobs is not None else None # [ K, N ]
        # parameters
        self.mu    = self.obs.mean(1) / 10 + 1e-2
        self.beta  = torch.nn.Parameter(torch.Tensor([2]))
        self.alpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01))
        self.gamma = torch.nn.Parameter(torch.Tensor(self.K, self.d).uniform_(0, .01))

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:    index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam_t: a vector of lambda value at location k = 0, 1, ..., K [ K ]
        """
        # current time and the past 
        t      = torch.ones(_t, dtype=torch.int32) * _t   # [ t ]
        tp     = torch.arange(_t)                         # [ t ]
        # self-exciting effect
        kernel = self._exp_kernel(self.beta, t, tp)       # [ t ]
        Nt     = self.obs[:, :_t].clone()                 # [ K, t ]
        lam1   = torch.mm(self.alpha, Nt * kernel).sum(1) # [ K ]
        # externel feature effect
        lam2   = 0
        if self.extobs is not None and _t - self.d >= 0:
            Xt   = self.extobs[:, _t - self.d:_t]         # [ K, d ]
            lam2 = (Xt * self.gamma).sum(1)
        lam    = torch.nn.functional.softplus(self.mu + lam1 + lam2)
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
        loglik = (Nloglams - lams).sum()
        return loglik, lams

    def forward(self):
        """
        customized forward function
        """
        return self._log_likelihood()

    @staticmethod
    def _exp_kernel(beta, t, tp):
        """
        Args:
        - t, tp: time index [ n ]
        """
        return beta * torch.exp(- (t - tp) * beta)
        


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
        alpha = module.alpha.data
        beta  = module.beta.data
        module.alpha.data = torch.clamp(alpha, min=0.)
        module.beta.data  = torch.clamp(beta, min=0.)



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
        alpha = module.alpha.data
        module.alpha.data = alpha * self.proxmat
    
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
        


if __name__ == "__main__":

    torch.manual_seed(1)

    obs_outage, obs_temp = MAdataloader()

    print(obs_outage.shape)
    print(obs_temp.shape)

    model = TorchHawkes(d=10, obs=obs_outage, extobs=None)

    # train(model, niter=200, lr=1., log_interval=10)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(model.state_dict(), "saved_models/hawkes-upt-mu.pt")

    model.load_state_dict(torch.load("saved_models/hawkes-upt-mu.pt"))
    
    print(model.beta.detach().numpy())

    _, lams = model()
    lams    = lams.detach().numpy()
    
    locs = np.load("data/geolocation.npy")
    import matplotlib.pyplot as plt 

    # ---------------------------------------------------
    # plt.plot(lams.sum(0))
    # plt.plot(obs_outage.sum(0))
    # plt.legend(["prediction", "real"])
    # plt.show()

    # ---------------------------------------------------

    alpha = model.alpha.detach().numpy()
    
    plt.figure()

    thres = [ .3, .5 ]
    pairs = [ (k1, k2) for k1 in range(model.K) for k2 in range(model.K) 
        if alpha[k1, k2] > thres[0] and alpha[k1, k2] < thres[1] ]

    # spectral clustering
    from sklearn.cluster import spectral_clustering
    from scipy.sparse import csgraph
    from numpy import inf, NaN

    cmap = ["red", "yellow", "green", "black", "purple"]
    # adj  = np.zeros((model.K, model.K))
    # for k1, k2 in pairs:
    #     adj[k1, k2] = alpha[k1, k2]
    # lap  = csgraph.laplacian(alpha, normed=False)
    ls   = spectral_clustering(
        affinity=alpha,
        n_clusters=5, 
        assign_labels="kmeans",
        random_state=0)

    print(len(ls))

    plt.scatter(locs[:, 1], locs[:, 0], s=12, c=[ cmap[l] for l in ls ])

    xs = [ (locs[k1][1], locs[k2][1]) for k1, k2 in pairs ] 
    ys = [ (locs[k1][0], locs[k2][0]) for k1, k2 in pairs ]
    cs = [ alpha[k1, k2] for k1, k2 in pairs ]

    for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
        # print(i, len(cs), c, alpha.min(), alpha.max())
        w = (c - thres[0]) / (thres[1] - thres[0]) 
        plt.plot(x, y, linewidth=w/3, color='blue')
        
    plt.show()

    # ---------------------------------------------------
    # print(model.gamma.detach().numpy())
    # gamma = model.gamma.detach().numpy()
    # plt.figure()
    # # plt.scatter(locs[:, 1], locs[:, 0], c=gamma.sum(1), cmap="hot")
    # for k in range(model.K):
    #     plt.plot(gamma[k, :])
    # plt.show()


    