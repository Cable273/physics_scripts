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

f1 = np.load("./pxp,half_graph,neel_fidelity,18.npy")
f2 = np.load("./pxp,neel_fidelity,18.npy")
t = np.arange(0,20,0.01)
plt.plot(t,f1,label="PXP Half Graph")
plt.plot(t,f2,label="PXP")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$\textrm{PXP Hypercube, Neel Fidelity,}$ $N=18$")
plt.show()

