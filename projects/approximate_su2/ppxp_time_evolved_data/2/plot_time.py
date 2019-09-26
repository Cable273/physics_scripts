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
from System_Classes import unlocking_System,U1_system
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

t=np.load("./pxp,subcube,t,16.npy")
y1 = np.load("./pxp,subcube,time_evolved,16.npy")
y2 = np.load("./pxp,perm,time_evolved,16.npy")
y3 = np.load("./pxp,fsa,time_evolved,16.npy")

y4 = np.load("./pxp,ppxp,subcube,time_evolved,16.npy")
y5 = np.load("./pxp,ppxp,perm,time_evolved,16.npy")
y6 = np.load("./pxp,ppxp,fsa,time_evolved,16.npy")


plt.plot(t,y1,linestyle="--",label=r"Subcube, $PXP$")
plt.plot(t,y2,linestyle="--",label=r"Perm, $PXP$")
plt.plot(t,y3,linestyle="--",label=r"FSA, $PXP$")

plt.plot(t,y4,label=r"Subcube, $PXP + \lambda (ZPXP+PXPZ)$")
plt.plot(t,y5,label=r"Perm, $PXP + \lambda (ZPXP+PXPZ)$")
plt.plot(t,y6,label=r"FSA, $PXP + \lambda (ZPXP+PXPZ)$")

plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi_{approx}(t) \vert \psi_{exact}(t) \rangle \vert^2$")
plt.title(r"$\textrm{Time Evolved overlap}, N=16$")
plt.show()

