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



if __name__ == "__main__":
    # Unit-test

    # i         = torch.LongTensor([[0, 1, 2, 4, 5], [5, 4, 3, 1, 0]])
    # v         = torch.FloatTensor([1, 2, 3, 4, 5])
    # spr_torch = torch.sparse.FloatTensor(i, v, torch.Size([6, 6]))
    # print(spr_torch.to_dense())
    # print(sparse_2D_slicing(spr_torch, 2, 5, dim=0).to_dense())
    # print(sparse_2D_flatten(spr_torch).to_dense())

    i   = [[0, 1, 2, 4, 5], [5, 4, 3, 1, 0]]
    v   = [1, 2, 3, 4, 5]
    csr = sparse.csr_matrix((v, i), shape=[6, 6])
    # print(spr_torch.toarray())
    # print(spr_torch[2:5, :].toarray())
    # print(spr_torch.transpose().reshape(1, -1).toarray())
    print(csr2torch(csr))