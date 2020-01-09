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
def exp(Q,psi):
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

N=14
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].model = np.array([[0,1,2,0]])
Hp[0].model_coef = np.array([1])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].model = np.array([[0,1,2,0,0],[0,0,1,2,0]])
Hp[1].model_coef = np.array([1,1])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[0,4,0,1,2,0],[0,1,2,0,4,0]])
Hp[2].model_coef = np.array([1,1])

psi=bin_state(np.array([0,1,0,1,1,0,0,1,0,1,1,0,0,0]),pxp)
k=pxp_syms.find_k_ref(psi.ref)

for n in range(0,len(Hp)):
    for m in range(0,np.size(k,axis=0)):
        Hp[n].gen(k[m])

coef = np.load("./data/xy,pert_coef,10.npy")
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))

Hp = Hp_total
Hm = Hp_total.herm_conj()
    
H = H_operations.add(Hp,Hm,np.array([1,1]))
for m in range(0,np.size(k,axis=0)):
    H.sector.find_eig(k[m])
    eig_overlap(psi,H,k[m]).plot()
plt.show()
fidelity(psi,H,"use sym").plot(np.arange(0,20,0.01),psi)
plt.show()
