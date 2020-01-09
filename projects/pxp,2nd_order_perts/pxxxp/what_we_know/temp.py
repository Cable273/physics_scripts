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

x1 = np.load("./pxp,pxxxp,scar_e_diff,no_pert,22.npy")
x2 = np.load("./pxp,pxxxp,scar_e_diff,opt,22.npy")
x3 = np.load("./pxp,pxxxp,scar_e_diff,ppxp,22.npy")

plt.plot(x1,label="No Pert")
plt.plot(x2,label="Optimal PXXXP")
plt.plot(x3,label="Optimal PPXP")
plt.xlabel(r"$n$")
plt.ylabel(r"$\Delta E_n$")
plt.title(r"$PXP$ Scar Energy Spacing, N=22")
plt.legend()
plt.show()

