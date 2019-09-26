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

import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def W(N):
    choose = ncr(int(N/2),2)
    return 1/np.power(choose,0.5)*1/np.power(int(N/2)-1,0.5)*N/2


N=50
d=6
c=7

#form initial double hypercube hamiltonian
# L = int((N+d-1)/2)
# print(L)
L = N
S=L/2
m = np.arange(-S,S)
couplings= np.power(S*(S+1)-m*(m+1),0.5)
print(couplings)

#hypercube tight binding
H0 = np.diag(couplings,1)+np.diag(couplings,-1)

#reflect it for the second cube connected at (0,0)
# dim = np.size(H0,axis=0)
# H=np.zeros((2*L+1,2*L+1))
# zero_index = dim
# H[:dim,:dim] = H0
# H[zero_index-1:,zero_index-1:] =H[:zero_index,:zero_index] 

H=H0

plt.matshow(H)
plt.show()

e,u = np.linalg.eigh(H)
psi_energy = np.conj(u[0,:])
t=np.arange(0,20,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.show()

# #triangle extra
# #propogate upwards, generating new basis states in each layer connected to neighbouring pairs in layer below
root_loc = np.arange(int(d/2),int(np.size(H,axis=0)-d/2))
root_count = N
#max = N-1
no_layers = N-1
for layer_count in range(0,no_layers):
    dim0 = np.size(H,axis=0)
    dim_to_add = root_count - 1

    zero_right_padding = np.zeros((np.size(H,axis=0),dim_to_add))
    H = np.hstack((H,zero_right_padding))
    zero_bot_padding = np.zeros((dim_to_add,np.size(H,axis=1)))
    H = np.vstack((H,zero_bot_padding))

    new_root_loc = np.arange(dim0,np.size(H,axis=0))

    for n in range(0,np.size(root_loc,axis=0)):
        if n == 0:
            b = np.random.uniform(0,c)
            H[root_loc[n],new_root_loc[0]] = c
            H[new_root_loc[0],root_loc[n]] = c
        elif n == np.size(root_loc,axis=0)-1:
            b = np.random.uniform(0,c)
            H[root_loc[n],new_root_loc[int(np.size(new_root_loc,axis=0)-1)]] = c
            H[new_root_loc[int(np.size(new_root_loc,axis=0)-1)],root_loc[n]] = c
        else:
            b = np.random.uniform(0,c)
            H[root_loc[n],new_root_loc[n-1]] = c
            H[root_loc[n],new_root_loc[n]] = c

            b = np.random.uniform(0,c)
            H[new_root_loc[n-1],root_loc[n]] = c
            H[new_root_loc[n],root_loc[n]] = c
    root_loc = new_root_loc
    root_count = np.size(root_loc)

to_del = []
for n in range(0,np.size(H,axis=0)):
    if (H[:,n] == np.zeros(np.size(H,axis=0))).all() or (H[n,:] == np.zeros(np.size(H,axis=0))).all():
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    H = np.delete(H,to_del[n],axis=0)
    H = np.delete(H,to_del[n],axis=1)
    
plt.matshow(np.abs(H))
plt.show()

e,u = np.linalg.eigh(H)
pbar=ProgressBar()
# index = 0
pbar=ProgressBar()
for index in pbar(range(0,np.size(H,axis=0))):
# for index in pbar(range(0,L)):
    psi_energy = np.conj(u[index,:])
    # plt.scatter(e,np.log10(np.abs(psi_energy)**2))
    # plt.show()

    t=np.arange(0,10,0.1)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,e,t[n])
        f[n] = np.abs(np.vdot(evolved_state,psi_energy)**2)
    plt.plot(t,f,alpha=0.4)

# psi_energy = np.conj(u[int(L/2),:])
# t = np.arange(0,10,0.01)
# f = np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
# plt.plot(t,f,label="Pol",linewidth=1,color="green")

psi_energy = np.conj(u[0,:])
t = np.arange(0,10,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f,label="Neel",linewidth=3,color="red")


plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"Hypercube, Triangle Bath, dim(H)="+str(np.size(H,axis=0)))
plt.legend()
plt.show()

overlap = np.log10(np.abs(psi_energy)**2)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"Hypercube, Triangle Bath, dim(H)="+str(np.size(H,axis=0)))
plt.scatter(e,overlap)
plt.show()

print(level_stats(e).mean())
