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

# N = 10
# pxp = unlocking_System([0,1],"periodic",2,N)
# pxp.gen_basis()
# pxp_syms=model_sym_data(pxp,[translational(pxp)])

L=20 #length of hypercube arm chain
d = 20 #number of tight binding nodes to keep
L_large = 60
coupling_site = 3  #nodes from either side before stitching on a second hypercube

#create left hypercube tight binding arm of length d
S=L/2
m = np.arange(-S,S)
couplings = np.power(S*(S+1)-m*(m+1),0.5)
H_arm = np.diag(couplings,1)+np.diag(couplings,-1)
H_arm = H_arm[:d,:d]

#bot of matrix is first site of coupled hypercube chain
#top of matrix is coupled node between two hypercube chains
H_arm = np.flip(np.flip(H_arm,axis=0),axis=1)
coupling_between_chains = H_arm[1,0]
print(coupling_between_chains)
H_arm_block = H_arm[1:,1:]

# create larger cube tight binding
S_large = L_large /2
m = np.arange(-S_large,S_large)
couplings = np.power(S_large*(S_large+1)-m*(m+1),0.5)
H_large = np.diag(couplings,1)+np.diag(couplings,-1)

#full H
H = H_large
#pad 1 row/col with zeros for coupling node
H_large = np.pad(H_large,pad_width=1,mode='constant')
H_large = np.delete(H_large,0,axis=0)
H_large = np.delete(H_large,0,axis=1)

#hypercube leg
new_D = np.size(H_large,axis=0)+np.size(H_arm_block,axis=0)-1
D_orig = np.size(H_large,axis=0)
H_large_new = np.zeros((new_D,new_D))
H_large_new[:D_orig,:D_orig] = H_large
H_large_new[D_orig-1:,D_orig-1:] = H_arm_block

coupling_indices = [D_orig-1]

H_large = H_large_new

#pad 1 row/col with zeros for coupling node
H_large = np.pad(H_large,pad_width=1,mode='constant')
H_large = np.delete(H_large,0,axis=0)
H_large = np.delete(H_large,0,axis=1)

#hypercube leg
new_D = np.size(H_large,axis=0)+np.size(H_arm_block,axis=0)-1
D_orig = np.size(H_large,axis=0)
H_large_new = np.zeros((new_D,new_D))
H_large_new[:D_orig,:D_orig] = H_large
H_large_new[D_orig-1:,D_orig-1:] = H_arm_block
H_large = H_large_new

coupling_indices = np.append(coupling_indices,D_orig-1)
H_large[coupling_indices[0],coupling_site-1] = coupling_between_chains
H_large[coupling_site-1,coupling_indices[0]] = coupling_between_chains

H_large[coupling_indices[1],L-(coupling_site-1)] = coupling_between_chains
H_large[L-(coupling_site-1),coupling_indices[1]] = coupling_between_chains

plt.matshow(np.abs(H_large))
plt.show()

subcube_edge_index = L_large+1+d-2
# H_large[subcube_edge_index,subcube_edge_index] = 10
# plt.matshow(H_large)
# plt.show()

e,u = np.linalg.eigh(H_large)
pbar=ProgressBar()
for index in pbar(range(0,np.size(H,axis=1))):
# for index in pbar(range(0,1)):
    # z_energy = np.conj(u[subcube_edge_index,:])
    z_energy = np.conj(u[index,:])
    # overlap = np.log10(np.abs(z_energy)**2)
    # plt.scatter(e,overlap)
    # plt.show()

    t=np.arange(0,10,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[n])
        f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
    plt.plot(t,f)
plt.show()
