#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import arrow
import random
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from netpcvx import CvxpyNetPoissonProcess
from dataloader import MAdataloader
from utils import avg
from plot import plot_data_on_map

if __name__ == "__main__":
    
    obs_outage, obs_temp, kcoord = MAdataloader(K=30)

    obs_outage = avg(obs_outage, N=2) # [180, 371]
    obs_temp   = avg(obs_temp, N=2)   # [180, 371]

    cp_netp = CvxpyNetPoissonProcess(d=12, data=obs_outage, coords=kcoord)
    cp_netp.fit(tau=12, t=180)
    b0, b1  = cp_netp.save_solution(_dir="cvx_params")