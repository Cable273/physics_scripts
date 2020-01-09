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

x=np.load("./z2,Hz_var,16.npy")
y=np.load("./z4,Hz_variance,16.npy")
plt.plot(x,label=r"$\vert Z_2\rangle$")
plt.plot(y,label=r"$\vert Z4 \rangle$")
plt.xlabel(r"$n$")
plt.ylabel(r"$var(H^z)$")
plt.title(r"$H_x = PXP$, $H_z$ Variance, N=20")
plt.legend()
plt.show()

