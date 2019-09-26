#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sparse_linalg
import sys
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def find_hamming_sectors(state_bits,system):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,system.N+1):
        hamming_sectors[n] = []
    for n in range(0,system.dim):
        h = 0
        for m in range(0,system.N,1):
            if system.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],system.basis_refs[n])
    return hamming_sectors

import numpy as np
import scipy as sp
import math

N = 20
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H=spin_Hamiltonian(pxp,"x",pxp_syms)
z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
    eig_overlap(z,H,k[n]).plot()
    

x1=np.load("./subcube,e,20.npy_")
x2=np.load("./subcube,perm,e,20.npy")
x3=np.load("./subcube,fsa,e,20.npy")

y1=np.load("./subcube,overlap,20.npy_")
y2=np.load("./subcube,perm,overlap,20.npy")
y3=np.load("./subcube,fsa,overlap,20.npy")

def dell(x,y):
    to_del=[]
    for n in range(0,np.size(y,axis=0)):
        if y[n] <-5:
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        y=np.delete(y,to_del[n])
        x=np.delete(x,to_del[n])
    return x,y

x1,y1 = dell(x1,y1)
x2,y2 = dell(x2,y2)
x3,y3 = dell(x3,y3)

plt.scatter(x1,y1,marker="s",color="red",s=100,alpha=0.6,label="Subcube")
plt.scatter(x2,y2,marker="x",s=100,label="Perm")
plt.scatter(x3,y3,marker="D",s=100,color="green",alpha=0.6,label="FSA")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"Scar approximations, overlap with Neel state, $PXP$, N="+str(pxp.N))
plt.legend()
plt.show()

x1=np.load("./subcube,e,20.npy_")
x2=np.load("./subcube,perm,e,20.npy")
x3=np.load("./subcube,fsa,e,20.npy")

y1=np.load("./approx_qual,cube,20.npy")
y2=np.load("./approx_qual,perm,20.npy")
y3=np.load("./approx_qual,fsa,20.npy")

plt.scatter(x1,y1,marker="s",color="red",s=100,alpha=0.6,label="Subcube")
plt.scatter(x2,y2,marker="x",s=100,label="Perm")
plt.scatter(x3,y3,marker="D",s=100,color="green",alpha=0.6,label="FSA")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"Scar approximations, overlap with exact eigenstates, $PXP$, N="+str(pxp.N))
plt.legend()
plt.show()
