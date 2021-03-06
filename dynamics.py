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

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

#init system
N=21
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])


# s=1
# m=np.arange(-s,s)
# couplings = np.power(s*(s+1)-m*(m+1),0.5)
# Sp = np.diag(couplings,1)
# Sm = np.diag(couplings,-1)
# Sz = 1/2 * com(Sp,Sm)
# Sx = 1/2 *(Sp + Sm)
# Sy = 1/2j *(Sp - Sm)

# H = Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = np.dot(Sz,Sz)
# H.site_ops[2] = Sx
# H.model = np.array([[1,2,1]])
# H.model_coef = np.array([1])
# z=zm_state(2,2,pxp)
# k=pxp_syms.find_k_ref(z.ref)
# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])
    # eig_overlap(z,H,k[n]).plot()
# plt.show()
# fidelity(z,H,"use sym").plot(np.arange(0,40,0.01),z)
# plt.show()

# # Hp = Hamiltonian(pxp,pxp_syms)
# # Hp.site_ops[1] = np.dot(Sz,Sz)
# # Hp.site_ops[2] = Sp
# # Hp.site_ops[3] = Sm
# # Hp.model = np.array([[1,2,1],[1,3,1]])
# # Hp.model_coef = np.array([1,1])
# # Hp.uc_size = np.array([2,2])
# # Hp.uc_pos = np.array([1,0])
# # Hp.gen()
# # Hm = Hp.herm_conj()
# # Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())
# # e,u = np.linalg.eigh(Hz)
# # from Diagnostics import print_wf
# # print(e)
# # print(z.bits)
# # print_wf(u[:,0],pxp,1e-2)
