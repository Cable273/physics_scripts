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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
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

N = 14
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0=spin_Hamiltonian(pxp,"x",pxp_syms)
H0.gen()

Hp0 = Hamiltonian(pxp,pxp_syms)
Hp0.site_ops[1] = np.array([[0,1],[0,0]])
Hp0.model = np.array([[1]])
Hp0.model_coef = np.array([1])
Hp0.gen()

Hp1 = Hamiltonian(pxp,pxp_syms)
Hp1.site_ops[1] = np.array([[0,1],[0,0]])
Hp1.site_ops[2] = np.array([[0,0],[1,0]])
Hp1.model = np.array([[0,1,2,1,0]])
Hp1.model_coef = np.array([1])
Hp1.gen()

Hm0 = Hamiltonian(pxp,pxp_syms)
Hm0.site_ops[1] = np.array([[0,0],[1,0]])
Hm0.model = np.array([[1]])
Hm0.model_coef = np.array([1])
Hm0.gen()

Hm1 = Hamiltonian(pxp,pxp_syms)
Hm1.site_ops[1] = np.array([[0,1],[0,0]])
Hm1.site_ops[2] = np.array([[0,0],[1,0]])
Hm1.model = np.array([[0,2,1,2,0]])
Hm1.model_coef = np.array([1])
Hm1.gen()

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

coef_vals = np.arange(0,0.2,0.001)
F_vals = np.zeros(np.size(coef_vals))
pbar=ProgressBar()
for n in pbar(range(0,np.size(coef_vals,axis=0))):
    Hp = H_operations.add(Hp0,Hp1,np.array([1,coef_vals[n]]))
    Hm = H_operations.add(Hm0,Hm1,np.array([1,coef_vals[n]]))
    Hz = com(Hp.sector.matrix(),Hm.sector.matrix())
    diff = Hz - np.diag(np.diag(Hz))
    F = np.power(np.trace(np.dot(diff,np.conj(np.transpose(diff)))),0.5)
    F_vals[n] = F
plt.plot(coef_vals,F_vals)
plt.show()
