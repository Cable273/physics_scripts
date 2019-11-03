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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi) - exp(Q,psi)**2

N=6
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

#pick pair of basis states (0100..) to build H+ so H+ + H- = PXP + a * PXXXP
model_space = unlocking_System([0,1],"open",2,N)
model_space.gen_basis()

config_error_matrix = np.load("./pxp,pxxxp,config_error_matrix,"+str(N)+".npy")
min_indices = np.where(config_error_matrix == np.min(config_error_matrix))
min_n = min_indices[0]
min_m = min_indices[1]
for n in range(0,np.size(min_n,axis=0)):
    print("\n")
    print(config_error_matrix[min_n[n],min_m[n]])
    print(model_space.basis[min_n[n]],model_space.basis[min_m[n]])

    


