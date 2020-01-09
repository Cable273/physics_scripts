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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
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

N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp = pxp.U1_sector(2)
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,0],[1,0]])
H.site_ops[2] = np.array([[0,1],[0,0]])
H.model = np.array([[0,1,2,0],[0,2,1,0]])
H.model_coef = np.array([1,1])
H.gen()

# plt.matshow(np.abs(H.sector.matrix()))
# plt.show()

from Calculations import connected_comps,plot_adjacency_graph
basis_labels = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    basis_labels[n] = pxp.basis[n]
plot_adjacency_graph(np.abs(H.sector.matrix()),labels=basis_labels)
plt.show()
# cc = connected_comps(H) 
# cc.find_connected_components()
# sizes = np.zeros(len(cc.components))
# for n in range(0,len(cc.components)):
    # sizes[n] = np.size(cc.components[n])
# print(np.sort(np.unique(sizes)))
    
# pxp.basis_refs = np.sort(cc.largest_component())
# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# pxp.keys = dict()
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
    # print(pxp.basis[n])
    # pxp.keys[pxp.basis_refs[n]] = n
# pxp.dim = np.size(pxp.basis_refs)

# H = Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = np.array([[0,0],[1,0]])
# H.site_ops[2] = np.array([[0,1],[0,0]])
# H.model = np.array([[0,1,2,0],[0,2,1,0]])
# H.model_coef = np.array([1,1])
# H.gen()
