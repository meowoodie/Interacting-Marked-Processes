#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import random
import numpy as np
import torch.optim as optim
from scipy import sparse
from tqdm import tqdm

# customized utilities
import torchutils as ut

def train(model, T=4*24, niter=1000, lr=1e+3, log_interval=64):
    """training procedure for one epoch"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    loss_log  = []
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for _iter in range(niter):
        tau = random.choice(range(model.d, model.N - T))
        try:
            model.train()
            optimizer.zero_grad()           # init optimizer (set gradient to be zero)
            loglik, _ = model(tau, tau + T) # log-likelihood
            loss      = - loglik            # negative log-likelihood
            loss.backward()                 # gradient descent
            optimizer.step()                # update optimizer
            # print("[%s] Train iteration: %d\tNeg Loglik: %.3e" % (arrow.now(), _iter, loss.item()))
            # log training output
            loss_log.append(loss.item())
            if _iter % log_interval == 0:
                print("[%s] Train epoch: %d\tLoss: %.3e" % (arrow.now(), _iter / log_interval, sum(loss_log) / log_interval))
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
        loglik = [ 
            torch.dot(omega.squeeze(), torch.log(lam).squeeze()) - lam.sum() # - log omega_k !, which is constant
            for lam, omega in zip(lams, omegas) ]                            # ( T-d )
        loglik = torch.sum(torch.stack(loglik))
        return loglik, lams

    def forward(self, tau, t):
        """
        customized forward function
        """
        return self._log_likelihood(tau, t)



class UnitNormClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))
        

if __name__ == "__main__":
    # Facts about data set
    # - 34589 time units and 371 locations
    # - 23.90% time are non-zeros
    # - 1.47% data entries are non-zeros
    data     = np.load("data/maoutage.npy")[2000:4000, :]
    print(data.shape)
    N, K     = data.shape
    spr_data = sparse.csr_matrix(data)
    tnetp    = TorchNetPoissonProcess(d=4*6, K=K, N=N, data=spr_data)

    # train(tnetp)
    # print("[%s] saving model..." % arrow.now())
    # torch.save(tnetp.state_dict(), "saved_models/new-half-day.pt")



    import matplotlib.pyplot as plt

    tnetp.load_state_dict(torch.load("saved_models/new-half-day.pt"))
    beta0 = tnetp.beta0.data.numpy()[0]
    beta1 = tnetp.beta1.data.numpy()
    print(beta1.shape)

    locs = np.load("data/geolocation.npy")
    
    poi = 200
    lag = 0
    # plt.scatter(locs[:, 1], locs[:, 0], s=8, c=beta1[poi, :, lag], cmap="hot", vmin=beta1.min(), vmax=beta1.max())
    plt.scatter(locs[:, 1], locs[:, 0], s=8, c=beta0, cmap="hot", vmin=beta1.min(), vmax=beta1.max())
    plt.scatter(locs[poi, 1], locs[poi, 0], s=40, c="r")
    plt.show()

    plt.plot(beta1[300, 200])
    plt.show()
