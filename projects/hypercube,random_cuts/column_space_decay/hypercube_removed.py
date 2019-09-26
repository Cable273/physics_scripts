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
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
hamming_length = 0
for n in range(0,len(hamming_sectors)):
    if np.size(hamming_sectors[n])!=0:
        hamming_length = hamming_length + 1
    print("\n")
    for m in range(0,np.size(hamming_sectors[n],axis=0)):
        print(pxp.basis[pxp.keys[hamming_sectors[n][m]]])
        


# def gaussian(x,mu,sigma):
    # return 1/np.power(2*math.pi*np.power(sigma,2),0.5)*np.exp(-(x-mu)**2/(2*np.power(sigma,2)))

# # np.random.seed(1)
# x=np.arange(0,hamming_length)
# std = 1.1
# fraction=0.1

# dist=gaussian(x,pxp.N/2,std)
# #rescale for
# middle_hamming_size = np.size(hamming_sectors[int(hamming_length/2)])
# dist = dist/np.max(dist)*middle_hamming_size*fraction
# dist = np.flip(np.round(dist)).astype(int)
# print(dist)

# refs_to_remove = dict()
# for n in range(0,np.size(dist,axis=0)):
    # refs_to_remove[n] = np.random.choice(hamming_sectors[n],dist[n])

# indices_to_remove = []
# temp  =[]
# for n in range(1,len(refs_to_remove)-1):
    # for m in range(0,np.size(refs_to_remove[n],axis=0)):
        # indices_to_remove = np.append(indices_to_remove,pxp.keys[refs_to_remove[n][m]])
        # temp = np.append(temp,pxp.basis_refs[pxp.keys[refs_to_remove[n][m]]])
# to_del = np.sort(np.unique(indices_to_remove))

# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # pxp.basis_refs=np.delete(pxp.basis_refs,to_del[n])
# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
    # pxp.keys[pxp.basis_refs[n]] = n
# pxp.dim = np.size(pxp.basis_refs)
# print(pxp.dim)

# H=spin_Hamiltonian(pxp,"x")
# H.gen()
# H.sector.find_eig()
# pbar=ProgressBar()
# t=np.arange(0,20,0.1)
# for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    # z=ref_state(pxp.basis_refs[n],pxp)
    # f = fidelity(z,H).eval(t,z)
    # plt.plot(t,f,alpha=0.4)

# z=zm_state(2,1,pxp)
# f = fidelity(z,H).eval(t,z)
# plt.plot(t,f,linestyle="--",linewidth=3,color="blue")

# z1=zm_state(2,1,pxp,1)
# f = fidelity(z,H).eval(t,z1)
# plt.plot(t,f,linestyle="--",linewidth=3,color="green")

# plt.show()
