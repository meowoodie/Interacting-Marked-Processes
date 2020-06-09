#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import numpy as np
import torch.optim as optim
from scipy import sparse

# customized utilities
import torchutils as ut


def train(model, niter=100, lr=1e-2):
    """training procedure for one epoch"""
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for epoch in range(niter):
        model.train()
        optimizer.zero_grad()       # init optimizer (set gradient to be zero)
        loglik, _ = model()         # log-likelihood
        loss      = - loglik        # negative log-likelihood
        loss.backward()             # gradient descent
        optimizer.step()            # update optimizer
        print("[%s] Train epoch: %d\tNeg Loglik: %.3f" % (arrow.now(), epoch, loss.item()))

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
        self.beta  = torch.nn.Parameter(torch.FloatTensor(1, self.kappa).uniform_(0, 1))
    
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
        
    def _log_likelihood(self, T):
        """
        Log likelihood function at time `T`
        
        Args:
        - T:      index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        """
        lams   = [ self._lambda(t) for t in range(self.d, T) ]                        # lambda values from d to T ( T-d, [K] )
        omegas = [ torch.Tensor(self.obs[t, :].toarray()) for t in range(self.d, T) ] # omega values from d to T  ( T-d, [K] )
        loglik = [ 
            torch.dot(omega.squeeze(), torch.log(lam).squeeze()) - lam.sum()          # - log omega_k !, which is constant
            for lam, omega in zip(lams, omegas) ]                                     # ( T-d )
        loglik = torch.sum(torch.stack(loglik))
        return loglik, lams

    def forward(self):
        return self._log_likelihood(self.N)
        

if __name__ == "__main__":
    # Facts about data set
    # - 34589 time units and 371 locations
    # - 23.90% time are non-zeros
    # - 1.47% data entries are non-zeros
    data     = np.load("data/maoutage.npy")
    N, K     = data.shape
    spr_data = sparse.csr_matrix(data)
    tnetp    = TorchNetPoissonProcess(d=100, K=K, N=N, data=spr_data)

    train(tnetp)




    # def __init__(self, n_class, n_sample, n_feature):
    #     """
    #     Args:
    #     - n_class:  number of classes
    #     - n_sample: total number of samples in a single batch (including all classes)
    #     """
    #     super(RobustClassifierLayer, self).__init__()
    #     self.n_class, self.n_sample, self.n_feature = n_class, n_sample, n_feature
    #     self.cvxpylayer = self._cvxpylayer(n_class, n_sample)

    # def forward(self, X_tch, Q_tch, theta_tch):
    #     """
    #     customized forward function. 
    #     X_tch is a single batch of the input data and Q_tch is the empirical distribution obtained from  
    #     the labels of this batch.
    #     input:
    #     - X_tch: [batch_size, n_sample, n_feature]
    #     - Q_tch: [batch_size, n_class, n_sample]
    #     - theta_tch: [batch_size, n_class]
    #     output:
    #     - p_hat: [batch_size, n_class, n_sample]
    #     """
    #     C_tch     = self._wasserstein_distance(X_tch)        # [batch_size, n_sample, n_sample]
    #     gamma_hat = self.cvxpylayer(theta_tch, Q_tch, C_tch) # (n_class [batch_size, n_sample, n_sample])
    #     gamma_hat = torch.stack(gamma_hat, dim=1)            # [batch_size, n_class, n_sample, n_sample]
    #     p_hat     = gamma_hat.sum(dim=2)                     # [batch_size, n_class, n_sample]
    #     return p_hat

    # @staticmethod
    # def _wasserstein_distance(X):
    #     """
    #     the wasserstein distance for the input data via calculating the pairwise norm of two aribtrary 
    #     data points in the single batch of the input data, denoted as C here. 
    #     see reference below for pairwise distance calculation in torch:
    #     https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        
    #     input
    #     - X: [batch_size, n_sample, n_feature]
    #     output
    #     - C_tch: [batch_size, n_sample, n_sample]
    #     """
    #     C_tch = []
    #     for x in X.split(split_size=1):
    #         x      = torch.squeeze(x, dim=0)                  # [n_sample, n_feature]
    #         x_norm = (x**2).sum(dim=1).view(-1, 1)            # [n_sample, 1]
    #         y_t    = torch.transpose(x, 0, 1)                 # [n_feature, n_sample]
    #         y_norm = x_norm.view(1, -1)                       # [1, n_sample]
    #         dist   = x_norm + y_norm - 2.0 * torch.mm(x, y_t) # [n_sample, n_sample]
    #         # Ensure diagonal is zero if x=y
    #         dist   = dist - torch.diag(dist)                  # [n_sample, n_sample]
    #         dist   = torch.clamp(dist, min=0.0, max=np.inf)   # [n_sample, n_sample]
    #         C_tch.append(dist)                                
    #     C_tch = torch.stack(C_tch, dim=0)                     # [batch_size, n_sample, n_sample]
    #     return C_tch

    # @staticmethod
    # def _cvxpylayer(n_class, n_sample):
    #     """
    #     construct a cvxpylayer that solves a robust classification problem
    #     see reference below for the binary case: 
    #     http://papers.nips.cc/paper/8015-robust-hypothesis-testing-using-wasserstein-uncertainty-sets
    #     """
    #     # NOTE: 
    #     # cvxpy currently doesn't support N-dim variables, see discussion and solution below:
    #     # * how to build N-dim variables?
    #     #   https://github.com/cvxgrp/cvxpy/issues/198
    #     # * how to stack variables?
    #     #   https://stackoverflow.com/questions/45212926/how-to-stack-variables-together-in-cvxpy 
        
    #     # Variables   
    #     # - gamma_k: the joint probability distribution on Omega^2 with marginal distribution Q_k and p_k
    #     gamma = [ cp.Variable((n_sample, n_sample)) for k in range(n_class) ]
    #     # - p_k: the marginal distribution of class k [n_class, n_sample]
    #     p     = [ cp.sum(gamma[k], axis=0) for k in range(n_class) ] 
    #     p     = cp.vstack(p) 

    #     # Parameters (indirectly from input data)
    #     # - theta: the threshold of wasserstein distance for each class
    #     theta = cp.Parameter(n_class)
    #     # - Q: the empirical distribution of class k obtained from the input label
    #     Q     = cp.Parameter((n_class, n_sample))
    #     # - C: the pairwise distance between data points
    #     C     = cp.Parameter((n_sample, n_sample))

    #     # Constraints
    #     cons = [ g >= 0. for g in gamma ]
    #     for k in range(n_class):
    #         cons += [cp.sum(cp.multiply(gamma[k], C)) <= theta[k]]
    #         for l in range(n_sample):
    #             cons += [cp.sum(gamma[k], axis=1)[l] == Q[k, l]]

    #     # Problem setup
    #     # total variation loss
    #     obj   = cp.Maximize(cp.sum(cp.min(p, axis=0)))
    #     # cross entropy loss
    #     # obj  = cp.Minimize(cp.sum(- cp.sum(p * cp.log(p), axis=0)))
    #     prob = cp.Problem(obj, cons)
    #     assert prob.is_dpp()

    #     # return cvxpylayer with shape (n_class [batch_size, n_sample, n_sample])
    #     # stack operation ('torch.stack(gamma_hat, axis=1)') is needed for the output of this layer
    #     # to convert the output tensor into a normal shape, i.e., [batch_size, n_class, n_sample, n_sample]
    #     return CvxpyLayer(prob, parameters=[theta, Q, C], variables=gamma)