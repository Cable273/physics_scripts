#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = np.load("./pxxxp,bipartite,ls,N_vals.npy")
coef = np.load("./pxxxp,bipartite,ls,coef_scaling.npy")
plt.plot(N,coef)
plt.xlabel(r"$N$")
plt.ylabel(r"$\lambda$")
plt.title(r"$PXP, PXXXP$ Biparite $H^+$ decomp"+"\n"+"Optimal level spacing coefficient scaling")
plt.show()

