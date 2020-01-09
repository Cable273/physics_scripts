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

t=np.arange(0,20,0.01)
f_fid1st = np.load("./1stOrderFid/16/pcp,1stOrder,fidelity,16.npy")
f_fid2st = np.load("./2ndOrderFid/16/pcp,2ndOrder,fidelity,16.npy")
f_var1st = np.load("./1stOrderSubVar/16/pcp,1stOrder,fidelity,16.npy")
f_var2st = np.load("./2ndOrderSubVar/16/pcp,2ndOrder,fidelity,16.npy")
f0 = np.load("./0Order/16/pcp,0stOrder,fidelity,16.npy")

plt.plot(t,f0,label="No Pert",linewidth=2)
plt.plot(t,f_var1st,label="1st Order",linewidth=2,linestyle="-.")
plt.plot(t,f_var2st,label="2nd Order",linewidth=2)
plt.legend()
plt.title(r"$PCP$, Optimal subspace variance pertubations, $N=16$")
plt.show()

plt.plot(t,f0,label="No Pert",linewidth=2)
plt.plot(t,f_fid1st,label="1st Order",linewidth=2,linestyle="-.")
plt.plot(t,f_fid2st,label="2nd Order",linewidth=2)
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$PCP$, Optimal fidelity pertubations, $N=16$")
plt.show()
