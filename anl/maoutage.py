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
rootpath = "/Users/woodie/Desktop/maoutage"
print("[%s] reading data into data frame ..." % arrow.now())
for filename in tqdm(os.listdir(rootpath)):
    if filename.endswith(".h5"):
        f = h5py.File(os.path.join(rootpath, filename), "r")
        t = arrow.get(filename, "YYYYMMDDHHmm")
        outage = np.array(f["outage"]).tolist()
        lat    = np.array(f["lat"])
        lon    = np.array(f["lon"])
        loc    = np.stack([lat, lon], axis=1)
        # only keep complete data entry
        if len(outage) == 371: 
            data.append([t.timestamp] + outage)

np.save("data/geolocation.npy", loc)

print("[%s] sorting the list by their timestamp ..." % arrow.now())
data = np.array(data)
# data.view('i8').sort(order=['f0'], axis=0)
data = data[np.argsort(data[:, 0])]

print("[%s] constructing complete data matrix as a matlab data file ..." % arrow.now())
start_t    = data[0, 0]  # start time
end_t      = data[-1, 0] # end time
data[:, 0] = (data[:, 0] - start_t) / (15 * 60)     # every 15 mins per event
N          = int((end_t - start_t) / (15 * 60)) + 1 # total number of events
mat        = np.zeros((N, 371), np.int32)
for di, mi in tqdm(enumerate(data[:, 0])):
    mat[mi, :] = data[di, 1:]

np.save("data/maoutage.npy", mat)
