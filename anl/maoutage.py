#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import arrow
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# read data into list
data     = []
counter  = 0
rootpath = "/Users/woodie/Desktop/outagex/maoutagex-2018"
print("[%s] reading data into data frame ..." % arrow.now())
for filename in tqdm(os.listdir(rootpath)):
    if filename.endswith(".h5"):
        f = h5py.File(os.path.join(rootpath, filename), "r")
        t = arrow.get(filename, "YYYYMMDDHHmm")
        outage  = np.array(f["outage"])
        idx     = np.array(f["idx"])
        lat     = np.array(f["lat"])
        lon     = np.array(f["lon"])
        loc     = np.stack([lat, lon, idx], axis=1)
        # sort by index
        order   = idx.argsort()
        idx     = idx[order]
        outage  = outage[order]
        loc     = loc[order, :]
        # remove duplicate entries
        _, _idx = np.unique(idx, return_index=True)
        loc     = loc[_idx, :]
        outage  = outage[_idx].tolist()
        # only keep complete data entry
        if len(outage) == 351: 
            data.append([t.timestamp] + outage)

np.save("data/geolocation_351.npy", loc)

print("[%s] sorting the list by their timestamp ..." % arrow.now())
data = np.array(data)
print(data.shape)
data = data[np.argsort(data[:, 0])]
print(data[0, 0])
print(arrow.get(data[0, 0]))
print(data[-1, 0])
print(arrow.get(data[-1, 0]))

print("[%s] constructing complete data matrix as a matlab data file ..." % arrow.now())
start_t    = data[0, 0]  # start time
end_t      = data[-1, 0] # end time
data[:, 0] = (data[:, 0] - start_t) / (15 * 60)     # every 15 mins per event
N          = int((end_t - start_t) / (15 * 60)) + 1 # total number of events
mat        = np.zeros((N, 351), np.int32)
for di, mi in tqdm(enumerate(data[:, 0])):
    mat[mi, :] = data[di, 1:]

np.save("data/maoutage_2018.npy", mat)