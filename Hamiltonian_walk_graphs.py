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

N = 9
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H=spin_Hamiltonian(pxp,"x",pxp_syms)
# H=clock_Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = np.array([[0,1/2],[1/2,0]])
# H.site_ops[2] = np.array([[1,0],[0,0]])
# H.site_ops[3] = np.array([[0,0],[0,1]])
# H.model=np.array([[2,1,3],[3,1,2]])
# H.model_coef = np.array([1,1])
# # H.model_coef = np.array([1,0])
k=[0]
H.gen(k)
# H.gen()

# H0=spin_Hamiltonian(pxp,"x",pxp_syms)
# H0.gen(k)
# # H = H_operations.add(H0,H,np.array([1,1/2]))
# H = H_operations.add(H0,H,np.array([1,1]))

# all states
# basis_labels = dict()
# for n in range(0,np.size(pxp.basis,axis=0)):
    # basis_labels[n] = pxp.basis[n]
# plot_adjacency_graph(np.abs(H.sector.matrix()),labels=basis_labels)
# # plt.title(r"$P^0 X P^1 + P^1 X P^0$, N="+str(pxp.N))
# # plt.title(r"$P^0 X P^1$, N="+str(pxp.N))
# plt.show()

# k=0 sector
refs = pxp_syms.find_block_refs(k)
basis_labels = dict()
colors=range(20)
for n in range(0,np.size(refs,axis=0)):
    basis_labels[n] = pxp.basis[pxp.keys[refs[n]]]
plot_adjacency_graph(np.abs(H.sector.matrix(k)),labels=basis_labels)
plt.show()
