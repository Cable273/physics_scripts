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
pxp = unlocking_System([0,1],"periodic",2,10)
pxp.gen_basis()

#form hamming rep starting from Neel
# z=zm_state(2,1,pxp)
z=ref_state(2,pxp)
hamming_sectors = find_hamming_sectors(z.bits)

for n in range(0,np.size(hamming_sectors[2])):
    print(pxp.basis[pxp.keys[hamming_sectors[2][n]]])

V = Hamiltonian(pxp)
V.site_ops[1] = np.array([[0,0],[0,1]])
V.model = np.array([[1]])
V.model_coef = np.array([1])
V.gen()

# H0 = spin_Hamiltonian(pxp,"x")
# H0.gen()

def column_rate(column_index):
    N_column = np.size(hamming_sectors[column_index])
    rate_sum = 0
    for ref1 in hamming_sectors[column_index]:
        for ref2 in hamming_sectors[column_index]:
            temp = np.dot(V.sector.matrix(),ref_state(ref1,pxp).prod_basis())
            int_term1 = np.abs(np.sum(temp))
            if ref1 == ref2:
                rate_sum = rate_sum + (N_column-1) * int_term1**2
            else:
                # print("HOHo")
                temp = np.dot(V.sector.matrix(),ref_state(ref2,pxp).prod_basis())
                int_term2 = np.abs(np.sum(temp))
                rate_sum = rate_sum - int_term1*int_term2
    rate_sum = -2*rate_sum/N_column**2
    return rate_sum

hamming_labels = np.arange(0,len(hamming_sectors))
decay_rates = np.zeros(len(hamming_sectors))
pbar=ProgressBar()
for n in pbar(range(0,np.size(hamming_labels,axis=0))):
    decay_rates[n] = column_rate(n)
plt.plot(hamming_labels,decay_rates,marker="s")
plt.xlabel(r"$\vert \textrm{col} j_0 \rangle$")
plt.ylabel("R")
plt.title(r"$H = X + \lambda n n, \textrm{ Q walk, vertex disorder, column decay rates}, N=10$")
plt.show()
