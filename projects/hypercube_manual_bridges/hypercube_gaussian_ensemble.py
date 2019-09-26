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


d=6 #d/2 edges either side of neel chain not connected to bath
D=200 #size of Gaussian random bath
L=40 #chain size of orig hypercube
c=1 #scaling of gaussian values
bond_scale = 20
var = 2.2
no_couplings = 1

#form initial double hypercube hamiltonian
S=L/2
m = np.arange(-S,S)
couplings= np.power(S*(S+1)-m*(m+1),0.5)

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

# #gaussian random extra
root_loc = np.arange(int(d/2),int(np.size(H,axis=0)-d/2))
root_count = L-d
H_rand = np.zeros((D,D))
for n in range(0,np.size(H_rand,axis=0)):
    for m in range(0,n):
        coupling = np.random.normal(0,c)
        H_rand[n,m] = coupling
        H_rand[m,n] = coupling
    
dim = np.size(H,axis=0)

H_new = np.zeros((np.size(H,axis=0)+np.size(H_rand,axis=0),np.size(H,axis=1)+np.size(H_rand,axis=1)))
dim0 = np.size(H,axis=0)
H_new[:dim0,:dim0] = H
H_new[dim:,dim:] = H_rand

#couple spin chain to bath
def gauss(x,x0,var):
    return np.exp(-(x-x0)**2/(2*var**2))/np.power(2*math.pi*var**2,0.5)

root_length = np.size(root_loc)
mid = int(root_length/2)
no_coupling_dist = gauss(np.arange(0,root_length),mid,var)
no_coupling_dist = 1/no_coupling_dist[int(mid)]*no_coupling_dist
no_coupling_dist = no_coupling_dist * bond_scale
plt.plot(no_coupling_dist)
plt.show()
print(no_coupling_dist)
for n in range(0,np.size(root_loc,axis=0)):
    for m in range(0,no_couplings):
        coupling_loc = np.random.choice(np.arange(dim,np.size(H_new,axis=0)))
        coupling = np.random.normal(0,c)
        H_new[root_loc[n],coupling_loc] = coupling
        H_new[coupling_loc,root_loc[n]] = coupling

H = H_new
print((np.abs(H-np.conj(np.transpose(H)))<1e-5).all())
print("dim="+str(np.size(H,axis=0)))
    
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
    plt.plot(t,f,alpha=0.6)

psi_energy = np.conj(u[int(L/2),:])
t = np.arange(0,10,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f,label="Pol",linewidth=1,color="green")

psi_energy = np.conj(u[0,:])
t = np.arange(0,10,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f,label="Neel",linewidth=2,color="red")


plt.xlabel(r"$t$")
plt.title(r"Hypercube, Gaussian Bath, dim(H)="+str(np.size(H,axis=0)))
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.legend()
plt.show()

overlap = np.log10(np.abs(psi_energy)**2)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.title(r"Hypercube, Gaussian Bath, dim(H)="+str(np.size(H,axis=0)))
plt.scatter(e,overlap)
plt.show()

print(level_stats(e).mean())
