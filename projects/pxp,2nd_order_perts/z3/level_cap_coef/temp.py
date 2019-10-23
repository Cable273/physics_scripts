#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

x=np.load("./pxp,z3,2nd_order_pert,coef,level_spacing,near_fid,15.npy")
y=np.load("./pxp,z3,2nd_order_pert,coef,level_spacing,15.npy")
z=np.load("./pxp,z3,2nd_order_pert,2_terms,coef,level_spacing,cap,15.npy")
print(x)
for n in range(0,np.size(x,axis=0)):
    print("%.4f" % x[n] )
# print(y)
# print(z)

