#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
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

import numpy as np
import scipy as sp
import math
from Calculations import plot_adjacency_graph

def check_basis_config(basis_config,uc_size,allowed_occ):
    N = np.size(basis_config)
    pass_checks = 1
    for n in range(0,N):
        cell_occ = np.zeros(uc_size)
        for m in range(0,uc_size):
            cell_occ[m] = basis_config[int((n+m) % N)]
        if np.sum(cell_occ) > allowed_occ:
            pass_checks = 0
            break
    if pass_checks == 1:
        return True
    else:
        return False

# init system
N=5
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
print(pxp.dim)

uc_size = 3
allowed_occ = 2

new_basis = np.zeros(pxp.N)
pbar=ProgressBar()
print("Decimating basis")
for n in pbar(range(0,np.size(pxp.basis,axis=0))):
    if check_basis_config(pxp.basis[n],uc_size,allowed_occ) == True:
        new_basis = np.vstack((new_basis,pxp.basis[n]))

pxp.basis = new_basis
pxp.basis_refs = np.zeros(np.size(pxp.basis,axis=0))
pxp.keys = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    pxp.basis_refs[n] = bin_to_int_base_m(pxp.basis[n],pxp.base)
    pxp.keys[pxp.basis_refs[n]] = n
pxp.dim = np.size(pxp.basis,axis=0)

H = spin_Hamiltonian(pxp,"x")
H.gen()

# H0=spin_Hamiltonian(pxp,"x",pxp_syms)
# H0.gen(k)
# # H = H_operations.add(H0,H,np.array([1,1/2]))
# H = H_operations.add(H0,H,np.array([1,1]))

# all states
basis_labels = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    basis_labels[n] = pxp.basis[n]
plot_adjacency_graph(np.abs(H.sector.matrix()),labels=basis_labels)
# plot_adjacency_graph(np.abs(H.sector.matrix()))
# plt.title(r"$P^0 X P^1 + P^1 X P^0$, N="+str(pxp.N))
# plt.title(r"$P^0 X P^1$, N="+str(pxp.N))
plt.show()

# k=0 sector
# refs = pxp_syms.find_block_refs(k)
# basis_labels = dict()
# colors=range(20)
# for n in range(0,np.size(refs,axis=0)):
    # basis_labels[n] = pxp.basis[pxp.keys[refs[n]]]
# plot_adjacency_graph(np.abs(H.sector.matrix(k)),labels=basis_labels)
# plt.show()
