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

N=16
exact_energy = np.load("./pcp,1stOrder,energy,"+str(N)+".npy")
exact_overlap = np.load("./pcp,1stOrder,overlap,"+str(N)+".npy")
su3_energy = np.load("./pcp,1stOrder,fsa_energy,"+str(N)+".npy")
su3_overlap = np.log10(np.load("./pcp,1stOrder,fsa_overlap,"+str(N)+".npy"))

su3_exact_overlap = np.log10(np.load("./pcp,1stOrder,fsa_exact_overlap,"+str(N)+".npy"))
# print(su3_exact_overlap)

plt.scatter(exact_energy,exact_overlap)
plt.scatter(su3_energy,su3_overlap,marker="x",s=100,color="red")
plt.title(r"$PCP$, 1st Order Perts, Optimize subspace variance , $N=16$")
plt.show()

plt.plot(su3_energy,su3_exact_overlap,marker="s")
plt.show()

t=np.arange(0,20,0.01)
f=np.load("./pcp,1stOrder,fidelity,16.npy")
plt.plot(t,f)
plt.show()

