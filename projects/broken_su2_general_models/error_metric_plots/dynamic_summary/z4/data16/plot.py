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
tol = -15
exact_energy = np.load("./pxp,0th_order,e,"+str(N)+".npy")
exact_overlap = np.load("./pxp,0th_order,z4_overlap,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z4_fsa,0th_order,e,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z4_fsa,0th_order,z4_overlap,"+str(N)+".npy")
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
plt.ylabel(r"$\log(\vert \langle E  \vert Z_4 \rangle \vert^2)$")
plt.show()

exact_energy = np.load("./pxp,1st_order,e,"+str(N)+".npy")
exact_overlap = np.load("./pxp,1st_order,z4_overlap,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z4_fsa,1st_order,e,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z4_fsa,1st_order,z4_overlap,"+str(N)+".npy")
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
plt.ylabel(r"$\log(\vert \langle E  \vert Z_4 \rangle \vert^2)$")
plt.show()

exact_energy = np.load("./pxp,2nd_order,e,"+str(N)+".npy")
exact_overlap = np.load("./pxp,2nd_order,z4_overlap,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z4_fsa,2nd_order,e,"+str(N)+".npy")
fsa_overlap = np.load("./pxp,z4_fsa,2nd_order,z4_overlap,"+str(N)+".npy")
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
plt.ylabel(r"$\log(\vert \langle E  \vert Z_4 \rangle \vert^2)$")
plt.show()

t=np.arange(0,20,0.01)
f0 = np.load("./pxp,0th_order,z4_fidelity,"+str(N)+".npy")
f1 = np.load("./pxp,1st_order,z4_fidelity,"+str(N)+".npy")
f2 = np.load("./pxp,2nd_order,z4_fidelity,"+str(N)+".npy")
plt.plot(t,f0,linewidth=2,label=r"0th Order Perts")
plt.plot(t,f1,linewidth=2,linestyle="--",label=r"1st Order Perts")
plt.plot(t,f2,linewidth=2,label=r"2nd Order Perts")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle Z_4 \vert e^{-iHt} \vert Z_4 \rangle \vert^2$")
plt.title(r"$PXP$ Spin $1/2$, $N=$"+str(N))
plt.show()

exact_energy = np.load("./pxp,z4_fsa,2nd_order,e,full,"+str(N)+".npy")
exact_entropy = np.load("./pxp,z4_fsa,2nd_order,entropy,"+str(N)+".npy")
scar_energy = np.load("./pxp,z4_fsa,2nd_order,scar_energy,"+str(N)+".npy")
scar_entropy = np.load("./pxp,z4_fsa,2nd_order,scar_entropy,"+str(N)+".npy")
fsa_energy = np.load("./pxp,z4_fsa,2nd_order,scar_energy,"+str(N)+".npy")
fsa_entropy = np.load("./pxp,z4_fsa,2nd_order,fsa_entropy,"+str(N)+".npy")
plt.scatter(exact_energy,exact_entropy)
plt.scatter(scar_energy,scar_entropy,marker="D",color="orange",alpha=0.4,s=300,label="ED Scars")
plt.scatter(fsa_energy,fsa_entropy,marker="x",color="red",s=300,alpha = 0.8,label=r"$su(2)$ Ritz vectors")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.show()


