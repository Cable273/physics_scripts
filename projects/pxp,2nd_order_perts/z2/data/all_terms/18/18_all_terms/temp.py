#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

x=np.load("./pxp,z2,2nd_order_perts,fid_coef,18.npy")
for n in range(0,np.size(x,axis=0)):
    print("%.6f" % x[n])

# print("\n")
# y = np.loadtxt("./pxp,z2,2nd_order_perts,Hz_var,16")
# print("%.8f" % np.max(y))
