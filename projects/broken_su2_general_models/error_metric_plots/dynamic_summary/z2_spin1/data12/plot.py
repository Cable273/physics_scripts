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

N=12
tol = -8
exact_energy = np.load("./pxp,0th_order,e,"+str(N)+".npy")
exact_overlap = np.load("./pxp,0th_order,z2_overlap,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z2_fsa,0th_order,e,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z2_fsa,0th_order,z2_overlap,"+str(N)+".npy")
to_del=[]
for n in range(0,np.size(exact_overlap,axis=0)):
    if exact_overlap[n] < tol:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    exact_overlap=np.delete(exact_overlap,to_del[n])
    exact_energy=np.delete(exact_energy,to_del[n])
    
plt.scatter(exact_energy,exact_overlap)
plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle E  \vert Z_2 \rangle \vert^2)$")
plt.show()

exact_energy = np.load("./pxp,1st_order,e,"+str(N)+".npy")
exact_overlap = np.load("./pxp,1st_order,z2_overlap,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z2_fsa,1st_order,e,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z2_fsa,1st_order,z2_overlap,"+str(N)+".npy")
to_del=[]
for n in range(0,np.size(exact_overlap,axis=0)):
    if exact_overlap[n] <tol:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    exact_overlap=np.delete(exact_overlap,to_del[n])
    exact_energy=np.delete(exact_energy,to_del[n])
plt.scatter(exact_energy,exact_overlap)
plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle E  \vert Z_2 \rangle \vert^2)$")
plt.show()

exact_energy = np.load("./pxp,1st_order,e,ppxp,"+str(N)+".npy")
exact_overlap = np.load("./pxp,1st_order,z2_overlap,ppxp,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z2_fsa,1st_order,e,ppxp,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z2_fsa,1st_order,z2_overlap,ppxp,"+str(N)+".npy")
to_del=[]
for n in range(0,np.size(exact_overlap,axis=0)):
    if exact_overlap[n] <tol:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    exact_overlap=np.delete(exact_overlap,to_del[n])
    exact_energy=np.delete(exact_energy,to_del[n])
plt.scatter(exact_energy,exact_overlap)
plt.scatter(fsa_energy,fsa_overlap,marker="x",color="red",s=100)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle E  \vert Z_2 \rangle \vert^2)$")
plt.show()

t=np.arange(0,20,0.01)
f0 = np.load("./pxp,0th_order,z2_fidelity,"+str(N)+".npy")
f1 = np.load("./pxp,1st_order,z2_fidelity,"+str(N)+".npy")
f2 = np.load("./pxp,1st_order,z2_fidelity,ppxp,"+str(N)+".npy")
plt.plot(t,f0,linewidth=2,label=r"0th Order Perts")
plt.plot(t,f1,linewidth=2,linestyle="--",label=r"$PPX_{01}P+PX_{01}PP$")
plt.plot(t,f2,linewidth=2,label=r"$PPXP+PXPP$")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle Z_2 \vert e^{-iHt} \vert Z_2 \rangle \vert^2$")
plt.title(r"$PXP$ Spin $1$, $N=$"+str(N))
plt.show()
