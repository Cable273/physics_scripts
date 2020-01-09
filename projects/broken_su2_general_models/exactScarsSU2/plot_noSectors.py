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

N = np.arange(6,21,3)
no_sectors = np.zeros(np.size(N))
for n in range(0,np.size(N,axis=0)):
    no_sectors[n] = np.load("./data/no_sectors,"+str(N[n])+".npy")

largest_sector = np.zeros(np.size(N))
for n in range(0,np.size(N,axis=0)):
    largest_sector[n] = np.load("./data/largest_sector,"+str(N[n])+".npy")
    
plt.plot(N,no_sectors,marker="s")
plt.xlabel(r"$N$")
plt.ylabel(r"Number of Sectors")
plt.title(r"$PXP - V$ Exact SU(2) Embedding")
plt.show()

plt.plot(N,largest_sector,marker="s")
plt.xlabel(r"$N$")
plt.ylabel(r"Largest Sector Size")
plt.title(r"$PXP - V$ Exact SU(2) Embedding")
plt.show()
