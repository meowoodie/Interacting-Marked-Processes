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

if __name__ == "__main__":
    
    obs_outage, obs_temp = MAdataloader()

    obs_outage = avg(obs_outage, N=8) # [45, 371]
    obs_temp   = avg(obs_temp, N=8)   # [45, 371]

    obs_outage = obs_outage[:, :10]
    print(obs_outage.shape)

    cp_netp = CvxpyNetPoissonProcess(d=10, data=obs_outage)
    cp_netp.fit(tau=10, t=45)
    b0, b1  = cp_netp.save_solution(_dir="cvx_params")

    import matplotlib.pyplot as plt
    
    for i in range(10):
        plt.plot(b1[0, i])
        plt.show()

    
