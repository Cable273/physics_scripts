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


N = 10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

def perm_key(sector):
    return bin_to_int_base_m(sector,int(pxp.N/2+1))

sector_refs = dict()
from_sector = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    bits = pxp.basis[n]
    c1 = 0
    c2 = 0
    for m in range(0,np.size(bits,axis=0)):
        if bits[m] == 1:
            if m % 2 == 0:
                c1 = c1 + 1
            else:
                c2 = c2 + 1
    sector = np.array([c1,c2])
    if perm_key(sector) in sector_refs.keys():
        sector_refs[perm_key(sector)] = np.append(sector_refs[perm_key(sector)],pxp.basis_refs[n])
    else:
        sector_refs[perm_key(sector)] = [pxp.basis_refs[n]]

keys = list(sector_refs.keys())
sector_labels = np.zeros((int(pxp.N/2),2))
for n in range(0,np.size(sector_labels,axis=0)):
    sector_labels[n,0] = n+1
sector_labels = np.flip(sector_labels,axis=0)
sector_labels = np.vstack((sector_labels,np.array([0,0])))
temp = np.flip(np.flip(np.delete(sector_labels,np.size(sector_labels,axis=0)-1,axis=0),axis=1),axis=0)
sector_labels = np.vstack((sector_labels,temp))
    
basis = np.zeros(pxp.dim)
for n in range(0,np.size(sector_labels,axis=0)):
    temp_state = np.zeros(pxp.dim)
    refs = sector_refs[perm_key(sector_labels[n])]
    for m in range(0,np.size(refs,axis=0)):
        temp_state[pxp.keys[refs[m]]] = 1
    temp_state = temp_state/ np.power(np.vdot(temp_state,temp_state),0.5)
    basis = np.vstack((basis,temp_state))
basis = np.transpose(np.delete(basis,0,axis=0))
    
H=spin_Hamiltonian(pxp,"x")
H.gen()

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
e,u = np.linalg.eigh(H_rot)
psi_energy = np.conj(u[0,:])
t = np.arange(0,20,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.show()
