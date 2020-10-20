#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import arrow
import numpy as np
import scipy.io as sio

# read data into list
data     = []
counter  = 0
rootpath = "/Users/woodie/Desktop"
print("[%s] reading data into data frame ..." % arrow.now())

with open("%s/location_with_totalcustomers/ma_loc_correction.csv" % rootpath, "r") as f:
    data       = [ line.strip("\n").split(",") for line in f.readlines() ]
    n_customer = np.array([ int(d[7]) for d in data[1:] ])
    np.save("data/ncustomer_ma.npy", n_customer)
    print(n_customer)

with open("%s/location_with_totalcustomers/gis_state_ga_gp.csv" % rootpath, "r") as f:
    data       = [ line.strip("\n").split(",") for line in f.readlines() ]
    n_customer = np.array([ int(d[5]) for d in data[1:] ])
    np.save("data/ncustomer_ga.npy", n_customer)
    print(n_customer)

with open("%s/location_with_totalcustomers/gis_state_nc_duke.csv" % rootpath, "r") as f:
    data       = [ line.strip("\n").split(",") for line in f.readlines() ]
    n_customer = np.array([ int(d[5]) for d in data[1:] ])
    np.save("data/ncustomer_ncsc.npy", n_customer)
    print(n_customer)