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

pxp = unlocking_System([0],"periodic",2,10)
pxp.gen_basis()
pxp_syms= model_sym_data(pxp,[translational(pxp)])
H=spin_Hamiltonian(pxp,"x",pxp_syms)
# z=bin_state(np.array((1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0)),pxp)
z=zm_state(2,1,pxp)
H.gen()
H.sector.find_eig()
eig_overlap(z,H).plot()
plt.show()
# fidelity(z,H).plot(np.arange(0,20,0.01),z)
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# plt.title(r"PXP, Defected Neel Fidelity, N="+str(pxp.N))
# plt.show()
