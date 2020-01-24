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
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

#init system
N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

decompBasis = unlocking_System([0,1],"periodic",2,N)
decompBasis.gen_basis()

to_keep = np.zeros(pxp.dim)
decomp_ref = []
pbar=ProgressBar()
for n in pbar(range(0,np.size(decompBasis.basis,axis=0))):
    Hp = Hamiltonian(pxp)
    Hp.site_ops[1] = np.array([[0,0],[1,0]])
    Hp.site_ops[2] = np.array([[0,1],[0,0]])
    model = []
    for m in range(0,np.size(decompBasis.basis[n],axis=0)):
        if decompBasis.basis[n][m] == 1:
            model.append([0,1,0])
        else:
            model.append([0,2,0])
    Hp.model = model
    Hp.model_coef = np.ones(pxp.N)
    Hp.uc_size = pxp.N*np.ones(pxp.N)
    Hp.uc_pos = np.arange(0,pxp.N)

    Hp.gen()
    Hm = Hp.herm_conj()

    Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())
    Hx = 1/2 * (Hp.sector.matrix()+Hm.sector.matrix())
    Hy= 1/2j * (Hp.sector.matrix()-Hm.sector.matrix())
    C = np.dot(Hx,Hx) + np.dot(Hy,Hy) + np.dot(Hz,Hz)

    ec,uc = np.linalg.eigh(C)
    for m in range(0,np.size(ec,axis=0)):
        if np.abs(var(Hz,uc[:,0]))<1e-5:
            to_keep = np.vstack((to_keep,uc[:,m]))
            decomp_ref = np.append(decomp_ref,decompBasis.basis_refs[n])
to_keep = np.transpose(np.delete(to_keep,0,axis=0))
loc = np.where(np.abs(to_keep)<1e-5)
to_keep[loc[0],loc[1]] = 0
to_keep = np.unique(to_keep,axis=1)
print(np.shape(to_keep))

# from Diagnostics import print_wf
# for n in range(0,np.size(to_keep,axis=1)):
    # print("\n")
    # print_wf(to_keep[:,n],pxp,1e-2)
    # print(to_keep[:,n])
    

