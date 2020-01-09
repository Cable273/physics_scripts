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

#init system
N=10
pxp = unlocking_System([0,1],"periodic",2,N,)
pxp.gen_basis()
pxp_syms  = model_sym_data(pxp,[translational(pxp)])

#create Hamiltonian
J = 0.7
h = 1.4
g = 1
H = Hamiltonian(pxp,pxp_syms)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[1,0],[0,-1]])
H.model = np.array([[1,1],[1],[2]])
H.model_coef= np.array([J,h,g])
H.gen()
H.sector.find_eig()
z=ref_state(0,pxp)
eig_overlap(z,H).plot()
plt.show()
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()
