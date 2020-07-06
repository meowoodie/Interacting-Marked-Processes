#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import random
import numpy as np
import cvxpy as cp
import torch.optim as optim
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances

from dataloader import MAdataloader
from utils import avg

class NetObservation(object):
    """
    Observations used by Interacting Network Process
    """
    def __init__(self, d, obs, coords):
        """
        Args:
        - d:   memory depth
        - K:   number of locations
        - N:   number of observed time units
        - obs: N x K observation in sparse (scipy.csr_matrix) matrix
        """
        self.N, self.K = obs.shape
        self.d         = d
        self.obs       = obs
        self.coords    = coords

    def eta(self, omega):
        """
        Eta function

        Args:
        - omega: d x K observation in sparse (scipy.csr_matrix) matrix [ d, K ]
        Return:
        - mat:   eta matrix [ K, K + K x K x d ]
        """
        I_K = np.eye(self.K)                   # [ K, K ]
        vec = omega.transpose().reshape(1, -1) # [ 1, K x d ]
        mat = sparse.kron(I_K, vec)            # [ K, K x K x d ]
        mat = sparse.hstack([I_K, mat])        # [ K, K + K x K x d ]
        return mat

class CvxpyNetPoissonProcess(NetObservation):
    """
    PyTorch Module for Interacting Network Process
    """

    def __init__(self, d, data, coords):
        """
        Args:
        - d:      memory depth
        - data:   data matrix [ N, K ]
        - coords: coordinates [ K, 2 ]
        """
        NetObservation.__init__(self, d, data, coords)
        # variables
        # beta0       = cp.Variable(self.K)                                              # [ K ]
        # beta1       = [ cp.Variable(d) for k in range(self.K) for k in range(self.K) ] # [ K x K, d ]

        beta0       = cp.Variable(self.K, nonneg=True)                           # [ K ]
        beta1       = [ cp.Variable(d, nonneg=True)
            for k in range(self.K) for k in range(self.K) ]                      # [ K x K, d ]
        self.beta   = [ beta0 ] + beta1                                          # [ K ] + [ K x K, d ]
        self.var    = cp.hstack(self.beta) # cvxpy variable in a form of 1D vector [ K + K x K x d ]   

    def fit(self, tau, t):
        """
        construct and fit the cvx solver

        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)
        """
        print("[%s] consisting of %d variables (%d memory depth and %d locations)" % 
            (arrow.now(), self.var.shape[0], self.d, self.K))
        print("[%s] constructing cvx solver ..." % arrow.now())
        # construct cvx solver
        prob = self._cvxsolver(tau=tau, t=t)
        print("[%s] solving problem ..." % arrow.now())
        # solve problem
        prob.solve()
        print("[%s] the optimal value is %f" % (arrow.now(), prob.value))
    
    def save_solution(self, _dir):
        """
        save fitted results to local file
        """
        beta0 = self.beta[0].value
        beta1 = [ vec.value for vec in self.beta[1:] ]
        beta1 = np.stack(beta1, axis=0).reshape((self.K, self.K, self.d))
        np.save("%s/beta0.npy" % _dir, beta0)
        np.save("%s/beta1.npy" % _dir, beta1)
        print("[%s] the solution has been saved to /%s" % (arrow.now(), _dir))
        return beta0, beta1
    
    def _lambda(self, t):
        """
        Conditional intensity function at time `t`

        Args:
        - t:     index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam_t: a vector of lambda value at location k = 0, 1, ..., K [ K ]
        """
        assert t < self.N and t >= self.d, "invalid time index %d > %d or < %d." % (t, self.N, self.d)
        omega = self.obs[t-self.d:t, :]    # observation in the past d time units [ d, K ]
        lam_t = self.eta(omega) @ self.var # [ K, K + K x K x d ] x [ K + K x K x d ] = [ K ]
        return lam_t

    def _log_likelihood(self, tau, t):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)
        Return:
        - loglik: a scalar of log-likelihood
        """
        assert t - tau >= self.d, "invalid time index %d > %d or < %d." % (t, self.N, self.d)
        lams   = [ self._lambda(_t) for _t in range(tau, t) ] # lambda values from tau to t ( t - tau, [ K ] )
        omegas = [ self.obs[_t, :] for _t in range(tau, t) ]  # omega values from tau to t  ( t - tau, [ K ] )
        # log-likelihood function
        loglik = [ 
            cp.sum(cp.multiply(omega, cp.log(lam))) - cp.sum(lam)
            for lam, omega in zip(lams, omegas) ]             # ( t - tau )
        loglik = cp.sum(cp.hstack(loglik))
        return loglik
    
    def _cvxsolver(self, tau, t, smoothness=1e-1, proxk=5):
        """
        cvxpy solver for maximum likelihood estimation
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)
        Return:
        - prob:   cvx solver
        """
        loglik = self._log_likelihood(tau, t)

        # constraint 1: enforce variable to be non-negative
        # con1 = [ self.var >= 0. ]
        # constraint 2: smoothness
        con2 = [ cp.abs(vec[1:] - vec[:-1]) <= smoothness for vec in self.beta[1:] ]
        # constraint 3: monotonicity
        con3 = [ vec[1:] <= vec[:-1] for vec in self.beta[1:] ]
        # constraint 4: spatial proximity 
        mask = self._proximity_mask(self.coords, k=proxk)
        con4 = [ self.beta[1 + k0 * self.K + k1] == 0. 
            for k0 in range(self.K) for k1 in range(self.K) 
            if mask[k0, k1] == 0. ]
        # set of constraints
        cons = con2 + con3 + con4

        # objective: maximize log-likelihood
        obj  = cp.Maximize(loglik)
        prob = cp.Problem(obj, cons)
        return prob

    @staticmethod
    def _proximity_mask(coords, k):
        """
        Args:
        - coords: a list of coordinates for K locations [ K, 2 ]
        """
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr
        
        # pairwise distance matrix
        distmat = euclidean_distances(coords)                           # [K, K]
        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        proxmat = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return proxmat



