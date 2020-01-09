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
exact_energy = np.load("./pcp,2ndOrder,energy,"+str(N)+".npy")
exact_overlap = np.load("./pcp,2ndOrder,overlap,"+str(N)+".npy")
su3_energy = np.load("./pcp,2ndOrder,fsa_energy,"+str(N)+".npy")
su3_overlap = np.log10(np.load("./pcp,2ndOrder,fsa_overlap,"+str(N)+".npy"))

su3_exact_overlap = np.log10(np.load("./pcp,2ndOrder,fsa_exact_overlap,"+str(N)+".npy"))

plt.scatter(exact_energy,exact_overlap)
to_del=[]
for n in range(0,np.size(su3_overlap,axis=0)):
    if su3_overlap[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    su3_overlap=np.delete(su3_overlap,to_del[n])
    su3_energy=np.delete(su3_energy,to_del[n])
plt.scatter(su3_energy,su3_overlap,marker="x",s=100,color="red")
    
plt.title(r"$PCP$, 2nd Order Perts, Optimize subspace variance , $N=16$")
plt.show()

plt.plot(su3_energy,su3_exact_overlap,marker="s")
plt.show()

t=np.arange(0,20,0.01)
f=np.load("./pcp,2ndOrder,fidelity,16.npy")
plt.plot(t,f)
plt.show()

