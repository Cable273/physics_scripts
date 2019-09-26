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
from System_Classes import unlocking_System
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

def find_hamming_sectors(state_bits):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,pxp.N+1):
        hamming_sectors[n] = []
    for n in range(0,pxp.dim):
        h = 0
        for m in range(0,pxp.N,1):
            if pxp.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],pxp.basis_refs[n])
    return hamming_sectors

#init small hypercube
N=6
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])
print(pxp.dim)
print(pxp.basis)

# z=zm_state(2,1,pxp)
# hamming_sectors = find_hamming_sectors(z.bits)
# hamming_length = 0
# for n in range(0,len(hamming_sectors)):
    # if np.size(hamming_sectors[n])!=0:
        # hamming_length = hamming_length + 1
    # print("\n")
    # for m in range(0,np.size(hamming_sectors[n],axis=0)):
        # print(pxp.basis[pxp.keys[hamming_sectors[n][m]]])

H=spin_Hamiltonian(pxp,"x",pxp_syms)
z=zm_state(2,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
for n in range(0,np.size(k,axis=0)):
    H.gen(k[n])
    H.sector.find_eig(k[n])
# plt.matshow(np.abs(H.sector.matrix()))
# plt.show()
# z=ref_state(0,pxp)
fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
plt.show()
for n in range(0,np.size(k,axis=0)):
    eig_overlap(z,H,k[n]).plot()
# z_energy = H.sector.eigvectors()[pxp.keys[z.ref],:]
# overlap = np.log(np.abs(z_energy)**2)
# plt.scatter(H.sector.eigvalues(),overlap)
plt.show()

print(pxp.dim)

# #cube 1
# pxp_half = unlocking_System([0,1],"periodic",2,int(N/2))
# pxp_half.gen_basis()

# #pad states to get embedded hypercube nodes
# cube1_basis = np.zeros((pxp_half.dim,N))
# cube2_basis = np.zeros((pxp_half.dim,N))
# for n in range(0,np.size(pxp_half.basis_refs,axis=0)):
    # c=0
    # for m in range(0,N-1,2):
        # cube1_basis[n,m] = pxp_half.basis[n][c]
        # cube2_basis[n,m+1] = pxp_half.basis[n][c]
        # c=c+1

# cube1_basis_refs = np.zeros(np.size(cube1_basis,axis=0))
# cube2_basis_refs = np.zeros(np.size(cube2_basis,axis=0))
# for n in range(0,np.size(cube1_basis_refs,axis=0)):
    # cube1_basis_refs[n] = bin_to_int_base_m(cube1_basis[n],pxp.base)
    # cube2_basis_refs[n] = bin_to_int_base_m(cube2_basis[n],pxp.base)

# cube1_basis_refs = np.delete(cube1_basis_refs,0,axis=0)
# cube2_basis_refs = np.delete(cube2_basis_refs,0,axis=0)

# coupling_basis = []
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # if pxp.basis_refs[n] not in cube1_basis_refs and pxp.basis_refs[n] not in cube2_basis_refs:
        # coupling_basis = np.append(coupling_basis,pxp.basis_refs[n])

# new_basis_refs = np.append(cube1_basis_refs,coupling_basis)
# new_basis_refs = np.append(new_basis_refs,cube2_basis_refs)

# new_basis = np.zeros((pxp.dim,pxp.dim))
# for n in range(0,np.size(new_basis_refs,axis=0)):
    # temp = ref_state(new_basis_refs[n],pxp).prod_basis()
    # new_basis[:,n] = temp

# H_rot = np.dot(np.conj(np.transpose(new_basis)),np.dot(H.sector.matrix(),new_basis))

# print(np.size(cube1_basis_refs))
# print(np.size(coupling_basis))
# print(np.size(cube2_basis_refs))

# plt.matshow(np.abs(H_rot))
# plt.show()
