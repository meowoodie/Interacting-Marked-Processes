#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import arrow
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def preproc_weather(fk, rootpath, savepath):
    # read data into list
    data     = []
    times    = []
    counter  = 0
    print("[%s] reading data into data frame ..." % arrow.now())
    for foldername in tqdm(os.listdir(rootpath)):
        if foldername == ".DS_Store":
            continue
        for filename in os.listdir(os.path.join(rootpath, foldername)):
            if not filename.endswith(".h5"):
                continue
            # print(foldername, filename)
            f = h5py.File(os.path.join(rootpath, foldername, filename), "r")
            t = arrow.get(foldername[:8] + filename[1:3] + "0000", "YYYYMMDDHHmm")
            # keys  = list(f.keys())
            # featk = keys[:131]
            # feat  = np.stack([ np.array(f[k]) for k in featk ], axis=1) # [ nlocations, nfeat ]
            feat  = np.array(f[fk])              # [ nlocations ]
            lat   = np.array(f["lat"])           # [ nlocations ]
            lon   = np.array(f["lon"])           # [ nlocations ]
            loc   = np.stack([lat, lon], axis=1) # [ nlocations, 2 ]
            times.append(t.timestamp)
            data.append(feat)

    # np.save("data/weathergeolocations.npy", loc)

    print("[%s] sorting the list by their timestamp ..." % arrow.now())
    data  = np.stack(data, axis=0)
    # print(data.shape)
    data  = data[np.argsort(times)]
    times = np.array(times)
    times = times[np.argsort(times)]
    print(data.shape)

    print("[%s] constructing complete data matrix as a data file ..." % arrow.now())
    start_t = times[0]  # start time
    end_t   = times[-1] # end time
    inds    = (times - start_t) / (60 * 60)     # every 60 mins per event
    N       = int((end_t - start_t) / (60 * 60)) + 1 # total number of events
    K       = data.shape[1]
    mat     = np.zeros((N, K), np.float32)
    for di, mi in tqdm(enumerate(inds)):
        mat[int(mi), :] = data[di]
    print(mat.shape)

    np.save(savepath, mat)

if __name__ == "__main__":
    fk_list = [ "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017" ]

    for fk in fk_list:
        preproc_weather(
            fk=fk, 
            rootpath="/Users/woodie/Desktop/maweather/h5cache_MA_201803_p", 
            savepath="data/maweather-201803/maweather-201803-feat%s.npy" % fk)