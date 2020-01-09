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

def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi) - exp(Q,psi)**2
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

Hp = Hamiltonian(pxp)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,1,0],[0,2,0]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([2,2])
Hp.uc_pos = np.array([1,0])
Hp.gen()
Hp = Hp.sector.matrix()

z=zm_state(2,1,pxp,1)
fsa_basis = z.prod_basis()
current_state = fsa_basis
fsa_dim = pxp.N
for n in range(0,fsa_dim):
    next_state = np.dot(Hp,current_state)
    if np.abs(np.vdot(next_state,next_state))**2 > 1e-5:
        next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
        fsa_basis = np.vstack((fsa_basis,next_state))
        current_state = next_state
    else:
        break
fsa_basis = np.transpose(fsa_basis)

Hm = np.conj(np.transpose(Hp))


Hz = 1/2 * com(Hp,Hm)

Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
for n in range(0,np.size(Hz_exp,axis=0)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])
    print(Hz_exp[n],Hz_var[n])
