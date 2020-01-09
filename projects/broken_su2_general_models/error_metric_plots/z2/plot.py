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

errors = np.load("./pxp,z2,su2_errors,18.npy")
print(errors)
plt.plot(errors[:,0],marker="s",linewidth=2,label=r"$1-f_0$")
plt.plot(errors[:,1],marker="s",linewidth=2,label=r"$\sigma$")
plt.plot(errors[:,2],marker="s",linewidth=2,label=r"$max(var(H^z)_n)$")
plt.plot(errors[:,3],marker="s",linewidth=2,label=r"$K$")
plt.legend()
plt.xlabel("Pertubation Order")
plt.ylabel("Error Measure")
plt.grid(True,which="major")
plt.yscale("log")
plt.title(r"$PXP$ Spin $1/2$, $\vert Z_2 \rangle$ $SU(2)$ Subspace Errors, $N=18$")
plt.show()

