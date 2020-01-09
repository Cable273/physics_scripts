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
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

#init system
N=14
pxp = unlocking_System([0],"periodic",2,N,)
pxp.gen_basis()

#perm basis
def perm_key(n,m):
    return bin_to_int_base_m([n,m],int(pxp.N/2)+1)
permsectors = dict()
for n in range(0,np.size(pxp.basis,axis=0)):
    bits = pxp.basis[n]
    aOcc = 0
    bOcc = 0
    for m in range(0,pxp.N):
        if bits[m] == 1:
            if m % 2 == 0:
                aOcc += 1
            else:
                bOcc += 1
    key = perm_key(aOcc,bOcc)
    if key in list(permsectors.keys()):
        permsectors[key] += ref_state(pxp.basis_refs[n],pxp).prod_basis()
    else:
        permsectors[key] = ref_state(pxp.basis_refs[n],pxp).prod_basis()

keys = list(permsectors.keys())
perm_basis = np.zeros((pxp.dim,np.size(keys)))
for n in range(0,np.size(keys,axis=0)):
    perm_basis[:,n] = permsectors[keys[n]] / np.power(np.vdot(permsectors[keys[n]],permsectors[keys[n]]),0.5)

# create Hamiltonian
H = spin_Hamiltonian(pxp,"x")
H.gen()
z=zm_state(2,1,pxp,1)
H.sector.find_eig()

H_perm = np.dot(np.conj(np.transpose(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
z_perm = np.dot(np.conj(np.transpose(perm_basis)),z.prod_basis())
e,u = np.linalg.eigh(H_perm)
perm_overlap = np.log10(np.abs(np.dot(np.conj(np.transpose(u)),z_perm))**2)
plt.matshow(np.abs(H_perm))
plt.show()

# eig_overlap(z,H).plot()
# plt.scatter(e,perm_overlap,marker="x",color="red",s=100)
# plt.show()

# Hp = Hamiltonian(pxp)
# Hp.site_ops[1] = np.array([[0,0],[1,0]])
# Hp.site_ops[2] = np.array([[0,1],[0,0]])
# Hp.model = np.array([[0,1,0],[0,2,0]])
# Hp.model_coef = np.array([1,1])
# Hp.uc_size = np.array([2,2])
# Hp.uc_pos = np.array([1,0])
# Hp.gen()
# Hm = Hp.herm_conj()
# Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())

# HzPerm0 = np.dot(np.conj(np.transpose(perm_basis)),np.dot(Hz,perm_basis))
# HpPerm = np.dot(np.conj(np.transpose(perm_basis)),np.dot(Hp.sector.matrix(),perm_basis))
# HmPerm = np.conj(np.transpose(HpPerm))
# HzPerm = 1/2 * com(HpPerm,HmPerm)
# print(np.diag(HzPerm0))
# print(np.diag(HzPerm))
# error1 = com(HzPerm0,HpPerm)-HpPerm
# error2 = com(HzPerm,HpPerm)-HpPerm
# error1 = np.sum(np.abs(error1)**2)
# error2 = np.sum(np.abs(error2)**2)
# print(error1)
# print(error2)
# plt.matshow(HP)


# from Calculations import gen_fsa_basis
# fsa_basis_perm = gen_fsa_basis(HpPerm,z_perm,pxp.N)
# #rotate back to regular basis
# fsaBasis = np.dot(perm_basis,fsa_basis_perm)
# from Diagnostics import print_wf
# for n in range(0,np.size(fsaBasis,axis=1)):
    # print("\n")
    # print_wf(fsaBasis[:,n],pxp,1e-2)
