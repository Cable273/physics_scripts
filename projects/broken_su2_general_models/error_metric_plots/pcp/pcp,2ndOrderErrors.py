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
from copy import deepcopy

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT
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
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=14
pxp = unlocking_System([0],"periodic",3,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2),PT(pxp)])

#orig H
Ip = dict()
Ip[0] = Hamiltonian(pxp,pxp_syms)
Ip[0].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[0].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[0].model = np.array([[0,1,0],[0,2,0]])
Ip[0].model_coef = np.array([1,-1])
Ip[0].uc_size = np.array([2,2])
Ip[0].uc_pos = np.array([1,0])

Ip[1] = Hamiltonian(pxp,pxp_syms)
Ip[1].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[1].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[1].model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Ip[1].model_coef = np.array([1,1,-1,-1])
Ip[1].uc_size = np.array([2,2,2,2])
Ip[1].uc_pos = np.array([0,1,1,0])

Ip[2] = Hamiltonian(pxp,pxp_syms)
Ip[2].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[2].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[2].model = np.array([[0,0,2,0,0],[0,0,1,0,0]])
Ip[2].model_coef = np.array([1,-1])
Ip[2].uc_size = np.array([2,2])
Ip[2].uc_pos = np.array([1,0])

Ip[3] = Hamiltonian(pxp,pxp_syms)
Ip[3].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[3].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[3].model = np.array([[0,1,0,0,0],[0,0,0,1,0],[0,0,0,2,0],[0,2,0,0,0]])
Ip[3].model_coef = np.array([1,1,-1,-1])
Ip[3].uc_size = np.array([2,2,2,2])
Ip[3].uc_pos = np.array([1,1,0,0])

Ip[4] = Hamiltonian(pxp,pxp_syms)
Ip[4].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[4].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[4].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Ip[4].model = np.array([[0,3,0,2,0],[0,2,0,3,0],[0,3,0,1,0],[0,1,0,3,0]])
Ip[4].model_coef = np.array([1,1,-1,-1])
Ip[4].uc_size = np.array([2,2,2,2])
Ip[4].uc_pos = np.array([0,0,1,1])

Ip[5] = Hamiltonian(pxp,pxp_syms)
Ip[5].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[5].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[5].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Ip[5].model = np.array([[0,3,0,2,0,0],[0,2,0,3,0,0],[0,0,3,0,2,0],[0,0,2,0,3,0],[0,3,0,1,0,0],[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,1,0,3,0]])
Ip[5].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Ip[5].uc_size = np.array([2,2,2,2,2,2,2,2])
Ip[5].uc_pos = np.array([0,0,1,1,1,1,0,0])

Ip[6] = Hamiltonian(pxp,pxp_syms)
Ip[6].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[6].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[6].model = np.array([[0,1,0,2,0,1,0],[0,2,0,1,0,2,0]])
Ip[6].model_coef = np.array([1,-1])
Ip[6].uc_size = np.array([2,2])
Ip[6].uc_pos = np.array([1,0])

Ip[7] = Hamiltonian(pxp,pxp_syms)
Ip[7].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[7].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[7].model = np.array([[0,2,0,2,0,1,0],[0,1,0,2,0,2,0],[0,0,2,1,0,2,0],[0,2,0,1,2,0,0],[0,0,1,2,0,1,0],[0,1,0,2,1,0,0],[0,1,0,1,0,2,0],[0,2,0,1,0,1,0]])
Ip[7].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Ip[7].uc_size = np.array([2,2,2,2,2,2,2,2])
Ip[7].uc_pos = np.array([0,0,1,1,0,0,1,1])

Ip[8] = Hamiltonian(pxp,pxp_syms)
Ip[8].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[8].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[8].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,2,0,0,0],[0,0,0,2,0,0]])
Ip[8].model_coef = np.array([1,1,-1,-1])
Ip[8].uc_size = np.array([2,2,2,2])
Ip[8].uc_pos = np.array([0,1,1,0])

Ip[9] = Hamiltonian(pxp,pxp_syms)
Ip[9].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[9].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[9].model = np.array([[0,1,2,0,1,0],[0,1,0,2,1,0],[0,2,1,0,2,0],[0,2,0,1,2,0]])
Ip[9].model_coef = np.array([1,1,-1,-1])
Ip[9].uc_size = np.array([2,2,2,2])
Ip[9].uc_pos = np.array([1,0,0,1])

root3 = np.power(3,0.5)
Ip[10] = Hamiltonian(pxp,pxp_syms)
Ip[10].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[10].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[10].site_ops[4] = np.array([[1/(2*root3),0,0],[0,-1/(2*root3),0],[0,0,-2/(2*root3)]])
Ip[10].model = np.array([[0,4,0,2,0],[0,2,0,4,0],[0,4,0,1,0],[0,1,0,4,0]])
Ip[10].model_coef = np.array([1,1,-1,-1])
Ip[10].uc_size = np.array([2,2,2,2])
Ip[10].uc_pos = np.array([0,0,1,1])

Ip[11] = Hamiltonian(pxp,pxp_syms)
Ip[11].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Ip[11].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Ip[11].site_ops[4] = np.array([[1/(2*root3),0,0],[0,-1/(2*root3),0],[0,0,-2/(2*root3)]])
Ip[11].model = np.array([[0,4,0,2,0,0],[0,0,2,0,4,0],[0,4,0,1,0,0],[0,0,1,0,4,0]])
Ip[11].model_coef = np.array([1,1,-1,-1])
Ip[11].uc_size = np.array([2,2,2,2])
Ip[11].uc_pos = np.array([0,1,1,0])

Im = dict()
Im[0] = Hamiltonian(pxp,pxp_syms)
Im[0].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[0].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[0].model = np.array([[0,2,0],[0,1,0]])
Im[0].model_coef = np.array([1,-1])
Im[0].uc_size = np.array([2,2])
Im[0].uc_pos = np.array([1,0])

Im[1] = Hamiltonian(pxp,pxp_syms)
Im[1].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[1].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[1].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Im[1].model_coef = np.array([1,1,-1,-1])
Im[1].uc_size = np.array([2,2,2,2])
Im[1].uc_pos = np.array([0,1,1,0])

Im[2] = Hamiltonian(pxp,pxp_syms)
Im[2].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[2].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[2].model = np.array([[0,0,1,0,0],[0,0,2,0,0]])
Im[2].model_coef = np.array([1,-1])
Im[2].uc_size = np.array([2,2])
Im[2].uc_pos = np.array([1,0])

Im[3] = Hamiltonian(pxp,pxp_syms)
Im[3].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[3].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[3].model = np.array([[0,2,0,0,0],[0,0,0,2,0],[0,0,0,1,0],[0,1,0,0,0]])
Im[3].model_coef = np.array([1,1,-1,-1])
Im[3].uc_size = np.array([2,2,2,2])
Im[3].uc_pos = np.array([1,1,0,0])

Im[4] = Hamiltonian(pxp,pxp_syms)
Im[4].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[4].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[4].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Im[4].model = np.array([[0,3,0,1,0],[0,1,0,3,0],[0,3,0,2,0],[0,2,0,3,0]])
Im[4].model_coef = np.array([1,1,-1,-1])
Im[4].uc_size = np.array([2,2,2,2])
Im[4].uc_pos = np.array([0,0,1,1])

Im[5] = Hamiltonian(pxp,pxp_syms)
Im[5].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[5].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[5].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Im[5].model = np.array([[0,3,0,1,0,0],[0,1,0,3,0,0],[0,0,3,0,1,0],[0,0,1,0,3,0],[0,3,0,2,0,0],[0,2,0,3,0,0],[0,0,3,0,2,0],[0,0,2,0,3,0]])
Im[5].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Im[5].uc_size = np.array([2,2,2,2,2,2,2,2])
Im[5].uc_pos = np.array([0,0,1,1,1,1,0,0])

Im[6] = Hamiltonian(pxp,pxp_syms)
Im[6].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[6].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[6].model = np.array([[0,2,0,1,0,2,0],[0,1,0,2,0,1,0]])
Im[6].model_coef = np.array([1,-1])
Im[6].uc_size = np.array([2,2])
Im[6].uc_pos = np.array([1,0])

Im[7] = Hamiltonian(pxp,pxp_syms)
Im[7].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[7].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[7].model = np.array([[0,1,0,1,0,2,0],[0,2,0,1,0,1,0],[0,0,1,2,0,1,0],[0,1,0,2,1,0,0],[0,0,2,1,0,2,0],[0,2,0,1,2,0,0],[0,2,0,2,0,1,0],[0,1,0,2,0,2,0]])
Im[7].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Im[7].uc_size = np.array([2,2,2,2,2,2,2,2])
Im[7].uc_pos = np.array([0,0,1,1,0,0,1,1])

Im[8] = Hamiltonian(pxp,pxp_syms)
Im[8].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[8].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[8].model = np.array([[0,0,2,0,0,0],[0,0,0,2,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]])
Im[8].model_coef = np.array([1,1,-1,-1])
Im[8].uc_size = np.array([2,2,2,2])
Im[8].uc_pos = np.array([0,1,1,0])

Im[9] = Hamiltonian(pxp,pxp_syms)
Im[9].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[9].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[9].model = np.array([[0,2,1,0,2,0],[0,2,0,1,2,0],[0,1,2,0,1,0],[0,1,0,2,1,0]])
Im[9].model_coef = np.array([1,1,-1,-1])
Im[9].uc_size = np.array([2,2,2,2])
Im[9].uc_pos = np.array([1,0,0,1])

root3 = np.power(3,0.5)
Im[10] = Hamiltonian(pxp,pxp_syms)
Im[10].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[10].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[10].site_ops[4] = np.array([[1/(2*root3),0,0],[0,-1/(2*root3),0],[0,0,-2/(2*root3)]])
Im[10].model = np.array([[0,4,0,1,0],[0,1,0,4,0],[0,4,0,2,0],[0,2,0,4,0]])
Im[10].model_coef = np.array([1,1,-1,-1])
Im[10].uc_size = np.array([2,2,2,2])
Im[10].uc_pos = np.array([0,0,1,1])

Im[11] = Hamiltonian(pxp,pxp_syms)
Im[11].site_ops[1] = np.array([[0,1,0],[0,0,0],[0,0,0]])
Im[11].site_ops[2] = np.array([[0,0,0],[1,0,0],[0,0,0]])
Im[11].site_ops[4] = np.array([[1/(2*root3),0,0],[0,-1/(2*root3),0],[0,0,-2/(2*root3)]])
Im[11].model = np.array([[0,4,0,1,0,0],[0,0,1,0,4,0],[0,4,0,2,0,0],[0,0,2,0,4,0]])
Im[11].model_coef = np.array([1,1,-1,-1])
Im[11].uc_size = np.array([2,2,2,2])
Im[11].uc_pos = np.array([0,1,1,0])

Kp = dict()
Kp[0] = Hamiltonian(pxp,pxp_syms)
Kp[0].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[0].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[0].model = np.array([[0,1,0],[0,2,0]])
Kp[0].model_coef = np.array([1,-1])
Kp[0].uc_size = np.array([2,2])
Kp[0].uc_pos = np.array([1,0])

Kp[1] = Hamiltonian(pxp,pxp_syms)
Kp[1].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[1].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[1].model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Kp[1].model_coef = np.array([1,1,-1,-1])
Kp[1].uc_size = np.array([2,2,2,2])
Kp[1].uc_pos = np.array([0,1,1,0])

Kp[2] = Hamiltonian(pxp,pxp_syms)
Kp[2].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[2].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[2].model = np.array([[0,0,2,0,0],[0,0,1,0,0]])
Kp[2].model_coef = np.array([1,-1])
Kp[2].uc_size = np.array([2,2])
Kp[2].uc_pos = np.array([1,0])

Kp[3] = Hamiltonian(pxp,pxp_syms)
Kp[3].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[3].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[3].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Kp[3].model = np.array([[0,3,0,1,0],[0,1,0,3,0],[0,3,0,2,0],[0,2,0,3,0]])
Kp[3].model_coef = np.array([1,1,-1,-1])
Kp[3].uc_size = np.array([2,2,2,2])
Kp[3].uc_pos = np.array([1,1,0,0])

Kp[4] = Hamiltonian(pxp,pxp_syms)
Kp[4].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[4].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[4].model = np.array([[0,1,0,0,0],[0,0,0,1,0],[0,2,0,0,0],[0,0,0,2,0]])
Kp[4].model_coef = np.array([1,1,-1,-1])
Kp[4].uc_size = np.array([2,2,2,2])
Kp[4].uc_pos = np.array([1,1,0,0])

Kp[5] = Hamiltonian(pxp,pxp_syms)
Kp[5].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[5].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[5].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Kp[5].model = np.array([[0,3,0,2,0,0],[0,0,2,0,3,0],[0,3,0,1,0,0],[0,0,1,0,3,0]])
Kp[5].model_coef = np.array([1,1,-1,-1])
Kp[5].uc_size = np.array([2,2,2,2])
Kp[5].uc_pos = np.array([0,1,1,0])

Kp[6] = Hamiltonian(pxp,pxp_syms)
Kp[6].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[6].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[6].model = np.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,2,0,0,0],[0,0,0,2,0,0]])
Kp[6].model_coef = np.array([1,1,-1,-1])
Kp[6].uc_size = np.array([2,2,2,2])
Kp[6].uc_pos = np.array([0,1,1,0])

Kp[7] = Hamiltonian(pxp,pxp_syms)
Kp[7].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[7].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[7].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Kp[7].model = np.array([[0,3,0,2,0],[0,2,0,3,0],[0,3,0,1,0],[0,1,0,3,0]])
Kp[7].model_coef = np.array([1,1,-1,-1])
Kp[7].uc_size = np.array([2,2,2,2])
Kp[7].uc_pos = np.array([0,0,1,1])

Kp[8] = Hamiltonian(pxp,pxp_syms)
Kp[8].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[8].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[8].model = np.array([[0,2,0,2,0,1,0],[0,2,0,1,2,0,0],[0,1,0,2,0,2,0],[0,0,2,1,0,2,0],[0,0,1,2,0,1,0],[0,2,0,1,0,1,0],[0,1,0,2,1,0,0],[0,1,0,1,0,2,0]])
Kp[8].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Kp[8].uc_size = np.array([2,2,2,2,2,2,2,2])
Kp[8].uc_pos = np.array([0,1,0,1,0,1,0,1,])

root3 = np.power(3,0.5)
Kp[9] = Hamiltonian(pxp,pxp_syms)
Kp[9].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[9].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[9].site_ops[4] = np.array([[1/(2*root3),0,0],[0,1/(2*root3),0],[0,0,-2/(2*root3)]])
Kp[9].model = np.array([[0,4,0,2,0],[0,2,0,4,0],[0,4,0,1,0],[0,1,0,4,0]])
Kp[9].model_coef = np.array([1,1,-1,-1])
Kp[9].uc_size = np.array([2,2,2,2])
Kp[9].uc_pos = np.array([0,0,1,1])

Kp[10] = Hamiltonian(pxp,pxp_syms)
Kp[10].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[10].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[10].site_ops[4] = np.array([[1/(2*root3),0,0],[0,1/(2*root3),0],[0,0,-2/(2*root3)]])
Kp[10].model = np.array([[0,4,0,2,0,0],[0,0,2,0,4,0],[0,2,0,4,0,0],[0,0,4,0,2,0],[0,4,0,1,0,0],[0,0,1,0,4,0],[0,0,4,0,1,0],[0,1,0,4,0,0]])
Kp[10].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Kp[10].uc_size = np.array([2,2,2,2,2,2,2,2])
Kp[10].uc_pos = np.array([0,1,0,1,1,0,0,1])

Kp[11] = Hamiltonian(pxp,pxp_syms)
Kp[11].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[11].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[11].model = np.array([[0,1,2,0,1,0],[0,1,0,2,1,0],[0,2,0,1,2,0],[0,2,1,0,2,0]])
Kp[11].model_coef = np.array([1,1,-1,-1])
Kp[11].uc_size = np.array([2,2,2,2])
Kp[11].uc_pos = np.array([1,0,1,0])

Kp[12] = Hamiltonian(pxp,pxp_syms)
Kp[12].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Kp[12].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Kp[12].model = np.array([[0,2,0,1,0,2,0],[0,1,0,2,0,1,0]])
Kp[12].model_coef = np.array([1,-1])
Kp[12].uc_size = np.array([2,2])
Kp[12].uc_pos = np.array([0,1])

Km = dict()
Km[0] = Hamiltonian(pxp,pxp_syms)
Km[0].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[0].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[0].model = np.array([[0,2,0],[0,1,0]])
Km[0].model_coef = np.array([1,-1])
Km[0].uc_size = np.array([2,2])
Km[0].uc_pos = np.array([1,0])

Km[1] = Hamiltonian(pxp,pxp_syms)
Km[1].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[1].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[1].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Km[1].model_coef = np.array([1,1,-1,-1])
Km[1].uc_size = np.array([2,2,2,2])
Km[1].uc_pos = np.array([0,1,1,0])

Km[2] = Hamiltonian(pxp,pxp_syms)
Km[2].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[2].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[2].model = np.array([[0,0,1,0,0],[0,0,2,0,0]])
Km[2].model_coef = np.array([1,-1])
Km[2].uc_size = np.array([2,2])
Km[2].uc_pos = np.array([1,0])

Km[3] = Hamiltonian(pxp,pxp_syms)
Km[3].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[3].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[3].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Km[3].model = np.array([[0,3,0,2,0],[0,2,0,3,0],[0,3,0,1,0],[0,1,0,3,0]])
Km[3].model_coef = np.array([1,1,-1,-1])
Km[3].uc_size = np.array([2,2,2,2])
Km[3].uc_pos = np.array([1,1,0,0])

Km[4] = Hamiltonian(pxp,pxp_syms)
Km[4].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[4].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[4].model = np.array([[0,2,0,0,0],[0,0,0,2,0],[0,1,0,0,0],[0,0,0,1,0]])
Km[4].model_coef = np.array([1,1,-1,-1])
Km[4].uc_size = np.array([2,2,2,2])
Km[4].uc_pos = np.array([1,1,0,0])

Km[5] = Hamiltonian(pxp,pxp_syms)
Km[5].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[5].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[5].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Km[5].model = np.array([[0,3,0,1,0,0],[0,0,1,0,3,0],[0,3,0,2,0,0],[0,0,2,0,3,0]])
Km[5].model_coef = np.array([1,1,-1,-1])
Km[5].uc_size = np.array([2,2,2,2])
Km[5].uc_pos = np.array([0,1,1,0])

Km[6] = Hamiltonian(pxp,pxp_syms)
Km[6].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[6].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[6].model = np.array([[0,0,2,0,0,0],[0,0,0,2,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]])
Km[6].model_coef = np.array([1,1,-1,-1])
Km[6].uc_size = np.array([2,2,2,2])
Km[6].uc_pos = np.array([0,1,1,0])

Km[7] = Hamiltonian(pxp,pxp_syms)
Km[7].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[7].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[7].site_ops[3] = np.array([[1/2,0,0],[0,-1/2,0],[0,0,0]])
Km[7].model = np.array([[0,3,0,1,0],[0,1,0,3,0],[0,3,0,2,0],[0,2,0,3,0]])
Km[7].model_coef = np.array([1,1,-1,-1])
Km[7].uc_size = np.array([2,2,2,2])
Km[7].uc_pos = np.array([0,0,1,1])

Km[8] = Hamiltonian(pxp,pxp_syms)
Km[8].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[8].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[8].model = np.array([[0,1,0,1,0,2,0],[0,1,0,2,1,0,0],[0,2,0,1,0,1,0],[0,0,1,2,0,1,0],[0,0,2,1,0,2,0],[0,1,0,2,0,2,0],[0,2,0,1,2,0,0],[0,2,0,2,0,1,0]])
Km[8].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Km[8].uc_size = np.array([2,2,2,2,2,2,2,2])
Km[8].uc_pos = np.array([0,1,0,1,0,1,0,1,])

root3 = np.power(3,0.5)
Km[9] = Hamiltonian(pxp,pxp_syms)
Km[9].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[9].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[9].site_ops[4] = np.array([[1/(2*root3),0,0],[0,1/(2*root3),0],[0,0,-2/(2*root3)]])
Km[9].model = np.array([[0,4,0,1,0],[0,1,0,4,0],[0,4,0,2,0],[0,2,0,4,0]])
Km[9].model_coef = np.array([1,1,-1,-1])
Km[9].uc_size = np.array([2,2,2,2])
Km[9].uc_pos = np.array([0,0,1,1])

Km[10] = Hamiltonian(pxp,pxp_syms)
Km[10].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[10].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[10].site_ops[4] = np.array([[1/(2*root3),0,0],[0,1/(2*root3),0],[0,0,-2/(2*root3)]])
Km[10].model = np.array([[0,4,0,1,0,0],[0,0,1,0,4,0],[0,1,0,4,0,0],[0,0,4,0,1,0],[0,4,0,2,0,0],[0,0,2,0,4,0],[0,0,4,0,2,0],[0,2,0,4,0,0]])
Km[10].model_coef = np.array([1,1,1,1,-1,-1,-1,-1])
Km[10].uc_size = np.array([2,2,2,2,2,2,2,2])
Km[10].uc_pos = np.array([0,1,0,1,1,0,0,1])

Km[11] = Hamiltonian(pxp,pxp_syms)
Km[11].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[11].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[11].model = np.array([[0,2,1,0,2,0],[0,2,0,1,2,0],[0,1,0,2,1,0],[0,1,2,0,1,0]])
Km[11].model_coef = np.array([1,1,-1,-1])
Km[11].uc_size = np.array([2,2,2,2])
Km[11].uc_pos = np.array([1,0,1,0])

Km[12] = Hamiltonian(pxp,pxp_syms)
Km[12].site_ops[1] = np.array([[0,0,1],[0,0,0],[0,0,0]])
Km[12].site_ops[2] = np.array([[0,0,0],[0,0,0],[1,0,0]])
Km[12].model = np.array([[0,1,0,2,0,1,0],[0,2,0,1,0,2,0]])
Km[12].model_coef = np.array([1,-1])
Km[12].uc_size = np.array([2,2])
Km[12].uc_pos = np.array([0,1])

Lp = dict()
Lp[0] = Hamiltonian(pxp,pxp_syms)
Lp[0].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp[0].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp[0].model = np.array([[0,1,0],[0,2,0]])
Lp[0].model_coef = np.array([1,-1])
Lp[0].uc_size = np.array([2,2])
Lp[0].uc_pos = np.array([1,0])

Lp[1] = Hamiltonian(pxp,pxp_syms)
Lp[1].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp[1].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp[1].model = np.array([[0,2,0,0],[0,0,2,0],[0,1,0,0],[0,0,1,0]])
Lp[1].model_coef = np.array([1,1,-1,-1])
Lp[1].uc_size = np.array([2,2,2,2])
Lp[1].uc_pos = np.array([0,1,1,0])

Lp[2] = Hamiltonian(pxp,pxp_syms)
Lp[2].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lp[2].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lp[2].model = np.array([[0,0,2,0,0],[0,0,1,0,0]])
Lp[2].model_coef = np.array([1,-1])
Lp[2].uc_size = np.array([2,2])
Lp[2].uc_pos = np.array([1,0])

Lm = dict()
Lm[0] = Hamiltonian(pxp,pxp_syms)
Lm[0].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lm[0].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lm[0].model = np.array([[0,2,0],[0,1,0]])
Lm[0].model_coef = np.array([1,-1])
Lm[0].uc_size = np.array([2,2])
Lm[0].uc_pos = np.array([1,0])

Lm[1] = Hamiltonian(pxp,pxp_syms)
Lm[1].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lm[1].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lm[1].model = np.array([[0,1,0,0],[0,0,1,0],[0,2,0,0],[0,0,2,0]])
Lm[1].model_coef = np.array([1,1,-1,-1])
Lm[1].uc_size = np.array([2,2,2,2])
Lm[1].uc_pos = np.array([0,1,1,0])

Lm[2] = Hamiltonian(pxp,pxp_syms)
Lm[2].site_ops[1] = np.array([[0,0,0],[0,0,1],[0,0,0]])
Lm[2].site_ops[2] = np.array([[0,0,0],[0,0,0],[0,1,0]])
Lm[2].model = np.array([[0,0,1,0,0],[0,0,2,0,0]])
Lm[2].model_coef = np.array([1,-1])
Lm[2].uc_size = np.array([2,2])
Lm[2].uc_pos = np.array([1,0])

k=[0,0]
z=zm_state(2,1,pxp)
for n in range(0,len(Ip)):
    Ip[n].gen(k)
for n in range(0,len(Im)):
    Im[n].gen(k)
for n in range(0,len(Kp)):
    print(n)
    Kp[n].gen(k)
for n in range(0,len(Km)):
    Km[n].gen(k)
for n in range(0,len(Lp)):
    Lp[n].gen(k)
for n in range(0,len(Lm)):
    Lm[n].gen(k)


def gen_su3Basis(coef):
    c=0
    Ip_total = deepcopy(Ip[0])
    for n in range(1,len(Ip)):
        Ip_total = H_operations.add(Ip_total,Ip[n],np.array([1,coef[c+n-1]]))
    Im_total = deepcopy(Im[0])
    for n in range(1,len(Im)):
        Im_total = H_operations.add(Im_total,Im[n],np.array([1,coef[c+n-1]]))
    c += len(Ip)-1
    Kp_total = deepcopy(Kp[0])
    for n in range(1,len(Kp)):
        Kp_total = H_operations.add(Kp_total,Kp[n],np.array([1,coef[c+n-1]]))
    Km_total = deepcopy(Km[0])
    for n in range(1,len(Km)):
        Km_total = H_operations.add(Km_total,Km[n],np.array([1,coef[c+n-1]]))
    c += len(Kp)-1
    Lp_total = deepcopy(Lp[0])
    for n in range(1,len(Lp)):
        Lp_total = H_operations.add(Lp_total,Lp[n],np.array([1,coef[c+n-1]]))
    Lm_total = deepcopy(Lm[0])
    for n in range(1,len(Lm)):
        Lm_total = H_operations.add(Lm_total,Lm[n],np.array([1,coef[c+n-1]]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))

    #su3 rep
    #0th order rep (no perts)
    root3 = np.power(3,0.5)
    I3 = 1/2 * com(Ip[0].sector.matrix(k),Im[0].sector.matrix(k))
    g8 = 1/(2*root3) * ( com(Kp[0].sector.matrix(k),Km[0].sector.matrix(k)) + com(Lp[0].sector.matrix(k),Lm[0].sector.matrix(k)) )

    def exp(Q,psi):
        return np.real(np.vdot(psi,np.dot(Q,psi)))
    def var(Q,psi):
        Q2 = np.dot(Q,Q)
        return exp(Q2,psi)-(exp(Q,psi))**2

    e,u = np.linalg.eigh(I3)
    lw = u[:,0]
    hw = u[:,np.size(u,axis=1)-1]
    #generate su3 representation by applying I+,L- to lw state
    su3_basis_states = dict()
    su3_basis = lw
    current_state = su3_basis
    i3_lw = exp(I3,lw)
    g8_hw = exp(g8,lw)
    current_S = np.abs(i3_lw)
    Ip_app= 0
    Lm_app= 0
    application_index = np.zeros(2)
    while np.abs(current_S)>1e-5:
        no_ip_apps = int(2*current_S)
        tagged_state = current_state
        for n in range(0,no_ip_apps):
            Ip_app = Ip_app + 1
            next_state = np.dot(Ip[0].sector.matrix(k),current_state)
            next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
            su3_basis = np.vstack((su3_basis,next_state))
            current_state = next_state
            application_index = np.vstack((application_index,np.array([Ip_app,Lm_app])))
        current_state = np.dot(Lm[0].sector.matrix(k),tagged_state)
        current_state = current_state / np.power(np.vdot(current_state,current_state),0.5)
        su3_basis = np.vstack((su3_basis,current_state))
        current_S = current_S - 1/2
        Ip_app = 0
        Lm_app += 1
        application_index = np.vstack((application_index,np.array([Ip_app,Lm_app])))
    su3_basis = np.transpose(su3_basis)

    #generate su3 dual representation (starting from highest weight state) representation by applying I-,L+ to lw state
    su3_basis_dual = hw
    current_state = su3_basis_dual
    i3_hw = exp(I3,hw)
    g8_lw = exp(g8,hw)
    current_S = np.abs(i3_hw)
    Im_app= 0
    Lp_app= 0
    application_index_dual = np.zeros(2)
    while np.abs(current_S)>1e-5:
        no_im_apps = int(2*current_S)
        tagged_state = current_state
        for n in range(0,no_im_apps):
            Im_app = Im_app + 1
            next_state = np.dot(Im[0].sector.matrix(k),current_state)
            next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
            su3_basis_dual = np.vstack((su3_basis_dual,next_state))
            current_state = next_state
            application_index_dual = np.vstack((application_index_dual,np.array([Im_app,Lp_app])))
        current_state = np.dot(Lp[0].sector.matrix(k),tagged_state)
        current_state = current_state / np.power(np.vdot(current_state,current_state),0.5)
        su3_basis_dual = np.vstack((su3_basis_dual,current_state))
        current_S = current_S - 1/2
        Im_app = 0
        Lp_app += 1
        application_index_dual = np.vstack((application_index_dual,np.array([Im_app,Lp_app])))
    su3_basis_dual = np.transpose(su3_basis_dual)

    su3_basis = np.hstack((su3_basis,su3_basis_dual))
    from Calculations import gram_schmidt
    gs = gram_schmidt(su3_basis)
    gs.ortho()
    su3_basis = gs.ortho_basis
    return su3_basis

def subspace_variance(coef):
    c=0
    Ip_total = deepcopy(Ip[0])
    for n in range(1,len(Ip)):
        Ip_total = H_operations.add(Ip_total,Ip[n],np.array([1,coef[c+n-1]]))
    Im_total = deepcopy(Im[0])
    for n in range(1,len(Im)):
        Im_total = H_operations.add(Im_total,Im[n],np.array([1,coef[c+n-1]]))
    c += len(Ip)-1
    Kp_total = deepcopy(Kp[0])
    for n in range(1,len(Kp)):
        Kp_total = H_operations.add(Kp_total,Kp[n],np.array([1,coef[c+n-1]]))
    Km_total = deepcopy(Km[0])
    for n in range(1,len(Km)):
        Km_total = H_operations.add(Km_total,Km[n],np.array([1,coef[c+n-1]]))
    c += len(Kp)-1
    Lp_total = deepcopy(Lp[0])
    for n in range(1,len(Lp)):
        Lp_total = H_operations.add(Lp_total,Lp[n],np.array([1,coef[c+n-1]]))
    Lm_total = deepcopy(Lm[0])
    for n in range(1,len(Lm)):
        Lm_total = H_operations.add(Lm_total,Lm[n],np.array([1,coef[c+n-1]]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))
    su3_basis = gen_su3Basis(coef)

    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H2,su3_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)
    return subspace_variance/np.size(su3_basis,axis=1)

def subspace_varianceSu2(coef):
    c=0
    Ip_total = deepcopy(Ip[0])
    for n in range(1,len(Ip)):
        Ip_total = H_operations.add(Ip_total,Ip[n],np.array([1,coef[c+n-1]]))
    Im_total = deepcopy(Im[0])
    for n in range(1,len(Im)):
        Im_total = H_operations.add(Im_total,Im[n],np.array([1,coef[c+n-1]]))
    c += len(Ip)-1
    Kp_total = deepcopy(Kp[0])
    for n in range(1,len(Kp)):
        Kp_total = H_operations.add(Kp_total,Kp[n],np.array([1,coef[c+n-1]]))
    Km_total = deepcopy(Km[0])
    for n in range(1,len(Km)):
        Km_total = H_operations.add(Km_total,Km[n],np.array([1,coef[c+n-1]]))
    c += len(Kp)-1
    Lp_total = deepcopy(Lp[0])
    for n in range(1,len(Lp)):
        Lp_total = H_operations.add(Lp_total,Lp[n],np.array([1,coef[c+n-1]]))
    Lm_total = deepcopy(Lm[0])
    for n in range(1,len(Lm)):
        Lm_total = H_operations.add(Lm_total,Lm[n],np.array([1,coef[c+n-1]]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))
    su3_basis = gen_su3Basis(coef)

    # restrict to 2N+1 basis with largest overlap with scar states
    # identify 2N+1 scars from H
    H.sector.find_eig(k)
    overlap = eig_overlap(z,H,k).eval()
    from Calculations import get_top_band_indices
    scar_indices = get_top_band_indices(H.sector.eigvalues(k),overlap,2*pxp.N,100,200,e_diff=0.5)
    plt.scatter(H.sector.eigvalues(k),overlap)
    for n in range(0,np.size(scar_indices,axis=0)):
        plt.scatter(H.sector.eigvalues(k)[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
    plt.show()
        

    #redifine su3_basis as the ritz vectors (new linear combs)
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    e,u = np.linalg.eigh(H_fsa)
    su3_basis = np.dot(su3_basis,u)

    #find 2N+1 basis states with largest overlap with scars states
    max_scar_overlap = np.zeros(np.size(su3_basis,axis=1))
    for n in range(0,np.size(max_scar_overlap,axis=0)):
        scarOverlap = np.zeros(np.size(scar_indices))
        for m in range(0,np.size(scarOverlap,axis=0)):
            scarOverlap[m] = np.vdot(su3_basis[:,n],H.sector.eigvectors(k)[:,scar_indices[m]])
        max_scar_overlap[n] = np.max(scarOverlap)

    su3_indices = np.arange(0,np.size(su3_basis,axis=1))
    max_scar_overlap,su3_indices = (list(t) for t in zip(*sorted(zip(max_scar_overlap,su3_indices))))
    max_scar_overlap = np.flip(max_scar_overlap)
    su3_indices = np.flip(su3_indices)
    su3_sub_indices = su3_indices[:np.size(scar_indices)]

    su3_sub_basis = np.zeros(np.size(su3_basis,axis=0))
    for n in range(0,np.size(su3_sub_indices,axis=0)):
        su3_sub_basis = np.vstack((su3_sub_basis,su3_basis[:,su3_sub_indices[n]]))
    su3_sub_basis = np.transpose(np.delete(su3_sub_basis,0,axis=0))
        
    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H2,su3_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_basis)),np.dot(H.sector.matrix(k),su3_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)

    H2 = np.dot(H.sector.matrix(k),H.sector.matrix(k))
    H2_fsa = np.dot(np.conj(np.transpose(su3_sub_basis)),np.dot(H2,su3_sub_basis))
    H_fsa = np.dot(np.conj(np.transpose(su3_sub_basis)),np.dot(H.sector.matrix(k),su3_sub_basis))
    subspace_variance = np.real(np.trace(H2_fsa-np.dot(H_fsa,H_fsa)))
    print(coef,subspace_variance)
    e,u = np.linalg.eigh(H_fsa)
    return subspace_variance/np.size(su3_sub_basis,axis=1)

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return f

from scipy.optimize import minimize_scalar
def fidelity_erorr(coef):
    c=0
    Ip_total = deepcopy(Ip[0])
    for n in range(1,len(Ip)):
        Ip_total = H_operations.add(Ip_total,Ip[n],np.array([1,coef[c+n-1]]))
    Im_total = deepcopy(Im[0])
    for n in range(1,len(Im)):
        Im_total = H_operations.add(Im_total,Im[n],np.array([1,coef[c+n-1]]))
    c += len(Ip)-1
    Kp_total = deepcopy(Kp[0])
    for n in range(1,len(Kp)):
        Kp_total = H_operations.add(Kp_total,Kp[n],np.array([1,coef[c+n-1]]))
    Km_total = deepcopy(Km[0])
    for n in range(1,len(Km)):
        Km_total = H_operations.add(Km_total,Km[n],np.array([1,coef[c+n-1]]))
    c += len(Kp)-1
    Lp_total = deepcopy(Lp[0])
    for n in range(1,len(Lp)):
        Lp_total = H_operations.add(Lp_total,Lp[n],np.array([1,coef[c+n-1]]))
    Lm_total = deepcopy(Lm[0])
    for n in range(1,len(Lm)):
        Lm_total = H_operations.add(Lm_total,Lm[n],np.array([1,coef[c+n-1]]))

    H=H_operations.add(Ip_total,Im_total,np.array([1j,-1j]))
    H=H_operations.add(H,Kp_total,np.array([1,-1j]))
    H=H_operations.add(H,Km_total,np.array([1,1j]))
    H=H_operations.add(H,Lp_total,np.array([1,1j]))
    H=H_operations.add(H,Lm_total,np.array([1,-1j]))

    H.sector.find_eig(k)

    z=zm_state(2,1,pxp,1)
    block_refs = pxp_syms.find_block_refs(k)
    psi = np.zeros(np.size(block_refs))
    loc = find_index_bisection(z.ref,block_refs)
    psi[loc] = 1
    psi_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors(k))),psi)

    t=np.arange(0,20,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(psi_energy,H.sector.eigvalues(k),t[n])
        f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f0 = np.max(f[cut:])
    plt.scatter(H.sector.eigvalues(k),np.log10(np.abs(psi_energy)**2))
    plt.title(r"$PCP+\lambda(PPCP+PCPP), N=$"+str(pxp.N))
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
    plt.show()
    plt.plot(t,f)
    plt.title(r"$PCP+\lambda(PPCP+PCPP), N=$"+str(pxp.N))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
    plt.show()
    return 1-f0
    
# from scipy.optimize import minimize
# res = minimize(lambda coef: subspace_variance(coef),method="powell",x0=[coef])
# plt.cla()
# subspace_variance([res.x])
# fidelity_erorr([res.x])

# subspace_varianceSu2([0])

coef = np.load("./data/2ndOrderFid/pcp,2ndOrderSU3,coef,fid,12.npy")
errors = np.zeros((2,3))
errors[0,0]= fidelity_erorr(coef)
errors[0,1]= subspace_variance(coef)
errors[0,2]= subspace_varianceSu2(coef)

coef = np.load("./data/2ndOrderSubVar/pcp,2ndOrderSU3,coef,12.npy")
errors[1,0]= fidelity_erorr(coef)
errors[1,1]= subspace_variance(coef)
errors[1,2]= subspace_varianceSu2(coef)

print(errors)
np.save("pcp,2ndOrderErrors,"+str(pxp.N),errors)


