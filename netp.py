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

# customized utilities
import torchutils as ut

def train(model, T=4*24, niter=1000, lr=1e-1, batch_size=10, log_interval=64):
    """training procedure for one epoch"""
    # coordinates of K locations
    coords    = np.load("data/geolocation.npy")[:, :2]
    # define model clipper to enforce inequality constraints
    clipper1  = NonNegativeClipper()
    clipper2  = ProximityClipper(coords, k=100)
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    logliklog = []
    reglog    = []
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _iter in range(niter):
        # taus = random.choices(range(model.d, model.N - T), k=batch_size)
        tau = random.choice(range(model.d, model.N - T))
        try:
            model.train()
            optimizer.zero_grad()           # init optimizer (set gradient to be zero)
            print(model.beta1[0])
            # logliks = [ model(tau, tau + T)[0] for tau in taus ]
            # logliks = torch.stack(logliks).mean()
            logliks = model(tau, tau + T)[0]
            reg     = torch.abs(model.beta1[:, :, 1:].clone() - model.beta1[:, :, :-1].clone()).sum()
            # objective function
            loss    = - logliks + 1e+6 * reg
            loss.backward()                 # gradient descent
            optimizer.step()                # update optimizer
            model.apply(clipper1)
            model.apply(clipper2)
            # log training output
            logliklog.append(logliks.item())
            reglog.append(reg.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train epoch: %d\tLoglik: %.3e\tReg: %.3e" % (
                    arrow.now(), 
                    _iter / log_interval, 
                    sum(logliklog) / log_interval, 
                    sum(reglog) / log_interval))
                loss_log = []
        except KeyboardInterrupt:
            break



class NetPoissonProcessObs(object):
    """
    Observations used by Interacting Network Process
    """
    def __init__(self, d, K, N, data):
        """
        Args:
        - d:    memory depth
        - K:    number of locations
        - N:    number of observed time units
        - data: N x K observation in sparse (scipy.csr_matrix) matrix
        """
        assert data.shape == (N, K), "incompatible observation input (data) with shape %s" % data.shape
        self.K   = K
        self.N   = N
        self.d   = d
        self.obs = data

    def eta(self, obs_tau2t):
        """
        Eta function

        Args:
        - obs_tau2t: d x K observation in sparse (scipy.csr_matrix) matrix [ d, K ]
        Return:
        - mat:       eta matrix [ d, d + K x K x d ]
        """
        I_K = np.eye(self.K)
        vec = obs_tau2t.transpose().reshape(1, -1) # [ 1, d x K ]
        mat = sparse.kron(I_K, vec)                # [ d, K x K x d ]
        mat = sparse.hstack([I_K, mat])            # [ d, d + K x K x d ]
        return mat



class TorchNetPoissonProcess(torch.nn.Module, NetPoissonProcessObs):
    """
    PyTorch Module for Interacting Network Process
    """

    def __init__(self, d, K, N, data):
        """
        """
        torch.nn.Module.__init__(self)
        NetPoissonProcessObs.__init__(self, d, K, N, data)
        self.K     = K
        self.d     = d
        self.kappa = K + K * K * d
        self.beta0 = torch.nn.Parameter(torch.FloatTensor(1, K).uniform_(0, 1))
        self.beta1 = torch.nn.Parameter(torch.FloatTensor(K, K, d).uniform_(0, 1))
        self.beta  = torch.cat([self.beta0, self.beta1.view(-1).unsqueeze(0)], dim=1)
    
    def _lambda(self, t):
        """
        Conditional intensity function at time `t`

        Args:
        - t:     index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam_t: a vector of lambda value at location k = 0, 1, ..., K [ K ]
        """
        assert t < self.N and t >= self.d, "invalid time index %d > %d or < %d." % (t, self.N, self.d)
        omg   = self.obs[t-self.d:t, :]       # observation in the past d time units
        eta   = ut.csr2torch(self.eta(omg))   # re-organize observations into eta matrix
        lam_t = torch.mm(eta, self.beta.t())  # compute conditionnal intensity at all locations
        return lam_t
        
    def _log_likelihood(self, tau, t):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)
        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   a list of historical conditional intensity values at time t = tau, ..., t
        """
        assert t - tau > self.d, "invalid time index %d > %d or < %d." % (t, self.N, self.d)
        lams   = [ self._lambda(_t) 
            for _t in range(tau, t) ]             # lambda values from tau to t ( t-tau, [K] )
        omegas = [ torch.Tensor(self.obs[_t, :].toarray()) 
            for _t in range(tau, t) ]             # omega values from tau to t  ( t-tau, [K] )
        # factorial term in log-likelihood function
        factos = []
        for omega in omegas:
            facto = omega.lgamma().exp()
            facto[facto == float('inf')] = 1e-10
            factos.append(facto)
        # log-likelihood function
        loglik = [ 
            torch.dot(omega.squeeze(), torch.log(lam).squeeze()) - lam.sum() - torch.log(facto).sum()
            for lam, omega, facto in zip(lams, omegas, factos) ] # ( T-d )
        loglik = torch.sum(torch.stack(loglik))
        return loglik, lams

    def forward(self, tau, t):
        """
        customized forward function
        """
        return self._log_likelihood(tau, t)



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
        beta1 = module.beta1.data
        beta0 = module.beta0.data
        module.beta1.data = torch.clamp(beta1, min=0.)
        module.beta0.data = torch.clamp(beta0, min=0.)



class ProximityClipper(object):
    """
    """

    def __init__(self, coords, k):
        """
        Args:
        - coords: a list of coordinates for K locations [ K, 2 ]
        """
        distmat      = euclidean_distances(coords)               # [K, K]
        proxmat      = self._k_nearest_mask(distmat, k=k)        # [K, K]
        self.proxmat = torch.FloatTensor(proxmat).unsqueeze_(-1) # [K, K, 1]
        
    def __call__(self, module):
        """enforce non-negative constraints"""
        beta1 = module.beta1.data
        mask  = self.proxmat.repeat(1, 1, module.d)
        module.beta1.data = beta1 * mask
    
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
    # random.seed(1)
    torch.manual_seed(1)

    # Facts about data set
    # - 34589 time units and 371 locations
    # - 23.90% time are non-zeros
    # - 1.47% data entries are non-zeros
    data     = np.load("data/maoutage.npy")[2000:4000, :]
    print(data.shape)
    N, K     = data.shape
    spr_data = sparse.csr_matrix(data)
    tnetp    = TorchNetPoissonProcess(d=4*6, K=K, N=N, data=spr_data)

    train(tnetp, niter=200, lr=1e-7, log_interval=1)
    print("[%s] saving model..." % arrow.now())
    torch.save(tnetp.state_dict(), "saved_models/upt-constrained-half-day.pt")



    import matplotlib.pyplot as plt

    # tnetp.load_state_dict(torch.load("saved_models/upt-constrained-half-day.pt"))
    beta0 = tnetp.beta0.data.numpy()[0]
    beta1 = tnetp.beta1.data.numpy()
    print(beta1.shape)

    locs = np.load("data/geolocation.npy")
    
    for poi in range(K):
        print(poi)
        lag = 0
        plt.figure()
        plt.scatter(locs[:, 1], locs[:, 0], s=8, c=beta1[poi, :, lag], cmap="hot", vmin=beta1.min(), vmax=beta1.max())
        # plt.scatter(locs[:, 1], locs[:, 0], s=8, c=beta0, cmap="hot", vmin=beta1.min(), vmax=beta1.max())
        plt.scatter(locs[poi, 1], locs[poi, 0], s=40, c="b")
        # plt.show()

        plt.savefig("imgs/scatter-%d.png" % poi)

        plt.figure()
        plt.plot(beta1[poi, poi])
        plt.savefig("imgs/line-%d.png" % poi)
