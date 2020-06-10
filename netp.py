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


def train(model, T=4*24*3, niter=1, nbatch=10, lr=1e+1):
    """training procedure for one epoch"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for epoch in range(niter):
        for idx in range(nbatch):
            tau = random.choice(range(model.d, model.N - T))
            print(tau)
            try:
                model.train()
                optimizer.zero_grad()           # init optimizer (set gradient to be zero)
                loglik, _ = model(tau, tau + T) # log-likelihood
                loss      = - loglik            # negative log-likelihood
                loss.backward()                 # gradient descent
                optimizer.step()                # update optimizer
                print("[%s] Train iteration: %d/%d\tNeg Loglik: %.3f" % (arrow.now(), epoch, idx, loss.item()))
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
        """
        assert t - tau > self.d, "invalid time index %d > %d or < %d." % (t, self.N, self.d)
        print("[%s] Evaluating conditional intensity..." % arrow.now())
        lams   = [ self._lambda(_t) for _t in tqdm(range(tau, t)) ]                   # lambda values from tau to t ( t-tau, [K] )
        omegas = [ torch.Tensor(self.obs[_t, :].toarray()) for _t in range(tau, t) ]  # omega values from tau to t  ( t-tau, [K] )
        print("[%s] Evaluating log likelihood..." % arrow.now())
        loglik = [ 
            torch.dot(omega.squeeze(), torch.log(lam).squeeze()) - lam.sum()          # - log omega_k !, which is constant
            for lam, omega in zip(lams, omegas) ]                               # ( T-d )
        loglik = torch.sum(torch.stack(loglik))
        return loglik, lams

    def forward(self, tau, t):
        return self._log_likelihood(tau, t)
        

if __name__ == "__main__":
    # Facts about data set
    # - 34589 time units and 371 locations
    # - 23.90% time are non-zeros
    # - 1.47% data entries are non-zeros
    data     = np.load("data/maoutage.npy")[2000:4000, :]
    print(data.shape)
    N, K     = data.shape
    spr_data = sparse.csr_matrix(data)
    tnetp    = TorchNetPoissonProcess(d=4*24, K=K, N=N, data=spr_data)

    train(tnetp)
    print("[%s] saving model..." % arrow.now())
    torch.save(tnetp.state_dict(), "saved_models/test.pt")

    # tnetp.load_state_dict(torch.load("saved_models/test.pt"))
    tnetp.beta
