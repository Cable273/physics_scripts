#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

x=np.loadtxt("./pxp,z3,2nd_order_perts,level_spacing,Hz_var,15")
for n in range(0,np.size(x,axis=0)):
    print(x[n])
    # print("%.4f" % x[n])


