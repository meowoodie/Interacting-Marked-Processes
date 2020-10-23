#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from scipy import sparse

def csr2torch(csr):
    """
    convert csr_matrix in scipy to sparse torch tensor
    """
    coo       = csr.tocoo()
    i         = torch.LongTensor(np.stack([coo.row, coo.col]))
    v         = torch.FloatTensor(coo.data)
    spr_torch = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
    return spr_torch

def sparse_2D_slicing(spr_torch, _from, _to, dim=0):
    """
    slice 2D sparse torch tensor from index `_from` to `_to` on `dim` dimension (dim={0, 1}).
    """
    i    = spr_torch._indices()                     # indices of nonzero entries [ 2, n ] (n is number of nonzero entries)
    v    = spr_torch._values()                      # values of nonzero entries  [ n ]
    size = spr_torch.shape                          # orginal size of input tensor
    pos  = (i[dim, :] >= _from) * (i[dim, :] < _to) # selected positions of indices after slicing
    # retrieve sliced new sparse tensor
    new_i          = i[:, pos]                      # new indices of nonzero entries
    new_i[dim, :] -= _from                          # shift indices to starting from 0
    new_v          = v[pos]                         # new values of nonzero entries
    new_size       = torch.Size([_to-_from, size[1]]) if dim == 0  else torch.Size([size[0], _to-_from])
    new_spr_torch  = torch.sparse.FloatTensor(new_i, new_v, new_size)
    return new_spr_torch

def sparse_2D_flatten(spr_torch):
    """
    flatten 2D sparse torch tensor.
    """
    i      = spr_torch._indices()                   # indices of nonzero entries
    v      = spr_torch._values()                    # values of nonzero entries
    nr, nc = spr_torch.shape                        # orginal size of input tensor
    # retrieve flatten new sparse tensor
    new_i         = i[0, :] * nc + i[1, :]          # new indices of nonzero entries
    new_i         = new_i.unsqueeze(0)
    new_size      = torch.Size([nr * nc])
    new_spr_torch = torch.sparse.FloatTensor(new_i, v, new_size)
    return new_spr_torch

def conv_covariates(rawX, model):
    """
    convolution of covariates with an exponential decaying kernel defined in model.
    """
    rawX   = torch.Tensor(rawX)
    conv_X = []
    for t in np.arange(rawX.shape[1]): # (model.N):
        # retrieve observed covariates in the past d time slots
        if t < model.d:
            x     = rawX[:, :t, :]                           # [ K, t, M ]
            x_pad = rawX[:, :1, :].repeat([1, model.d-t, 1]) # [ K, d - t, M ]
            x     = torch.cat([x_pad, x], dim=1)             # [ K, d, M ]
        else:
            x     = rawX[:, t-model.d:t, :]                  # [ K, d, M ]
        # convolution with an exponential decaying kernel
        conv_x    = model.conv_exp_decay_kernel(x)           # [ K, M ]
        # conv_x    = x.sum(1)                                 # [ K, M ]
        conv_X.append(conv_x)                                # ( N, [ K, M ])
    conv_X = torch.stack(conv_X, dim=1)                      # [ K, N, M ]
    return conv_X