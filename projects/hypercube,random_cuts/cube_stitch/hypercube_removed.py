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
N=8
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()

H=spin_Hamiltonian(pxp,"x")
H.gen()
H_total = np.zeros((2*pxp.dim,2*pxp.dim))
H_total[0:pxp.dim,0:pxp.dim] = H.sector.matrix()
H_total[pxp.dim:,pxp.dim:] = H.sector.matrix()
for m in range(0,np.size(H_total[pxp.dim,:],axis=0)):
    if H_total[pxp.dim,m] != 0:
        print("HOHo")
        H_total[0,m] = H_total[pxp.dim,m]
        H_total[m,0] = H_total[m,pxp.dim]
H_total = np.delete(H_total,pxp.dim,axis=0)
H_total = np.delete(H_total,pxp.dim,axis=1)
e,u = np.linalg.eigh(H_total)
print(np.size(e))
plt.matshow(H_total)
plt.show()

# z=zm_state(2,1,pxp)
# t=np.arange(0,20,0.01)
# pbar=ProgressBar()
# for index in pbar(range(0,np.size(pxp.basis_refs))):
    # z=ref_state(pxp.basis_refs[index],pxp)
    # z_energy =np.conj(u[pxp.keys[z.ref],:])
    # f=np.zeros(np.size(t))
    # for n in range(0,np.size(f,axis=0)):
        # evolved_state = time_evolve_state(z_energy,e,t[n])
        # f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
    # plt.plot(t,f)
# plt.show()

# # z=zm_state(2,1,pxp)
# # hamming_sectors = find_hamming_sectors(z.bits)
# # hamming_length = 0
# # for n in range(0,len(hamming_sectors)):
    # # if np.size(hamming_sectors[n])!=0:
        # # hamming_length = hamming_length + 1
    # # print("\n")
    # # for m in range(0,np.size(hamming_sectors[n],axis=0)):
        # # print(pxp.basis[pxp.keys[hamming_sectors[n][m]]])
        


