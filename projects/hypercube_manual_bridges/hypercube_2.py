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


L = 20
S=L/2
m = np.arange(-S,S)
couplings= np.power(S*(S+1)-m*(m+1),0.5)

#hypercube tight binding
H0 = np.diag(couplings,1)+np.diag(couplings,-1)

#reflect it for the second cube connected at (0,0)
dim = np.size(H0,axis=0)
H=np.zeros((2*L+1,2*L+1))
zero_index = dim
H[:dim,:dim] = H0
H[zero_index-1:,zero_index-1:] =H[:zero_index,:zero_index] 

plt.matshow(H)
plt.show()
psi = np.zeros(np.size(H,axis=0))
psi[0]=1
plt.plot(psi)
plt.show()

e,u = np.linalg.eigh(H)
psi_energy = np.conj(u[0,:])
t=np.arange(0,20,0.1)
f = np.zeros(np.size(t))
pbar=ProgressBar()
for n in pbar(range(0,np.size(t,axis=0))):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    # evolved_state_comp = np.dot(u,evolved_state)
    # plt.plot(np.abs(evolved_state_comp)**2)
    # plt.axhline(y=1)
    # plt.axvline(dim)
    # plt.xlabel(r"Hamming distance")
    # plt.ylabel(r"$\vert \psi(n) \vert^2$")
    # plt.title(r"2 Hypercubes, L="+str(L))
    # plt.tight_layout()
    # plt.savefig("./temp/img"+str(n))
    # plt.cla()
    # plt.show()
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"Two hypercubes connected at $\vert0000...\rangle$, Fidelity, $L_{hamming}$="+str(L))

plt.show()

