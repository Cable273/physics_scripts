#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

x=np.load("./pxp,z4,pert_coef.npy")
for n in range(0,np.size(x,axis=0)):
    print("%.4f" % x[n])

