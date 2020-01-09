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

class Hp_config:
    def __init__(self,pxp_config,pxxxp_config):
        self.pxp_config = pxp_config
        self.pxxxp_config = pxxxp_config

        res = minimize_scalar(lambda a: spacing_error(a,pxp_config,pxxxp_config),method="Golden",bracket=(-0.2,0.2))
        self.coef = res.x
        self.ls_error = res.fun

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
def exp(Q,psi):
    return np.vdot(psi,np.dot(Q,psi))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi) - exp(Q,psi)**2

N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

model_space = unlocking_System([0,1],"open",2,N)
model_space.gen_basis()

config_coef_matrix = np.load("./data/"+str(pxp.N)+"/pxp,pxxxp,config_coef_matrix,"+str(pxp.N)+".npy")
config_ls_error_matrix = np.load("./data/"+str(pxp.N)+"/pxp,pxxxp,config_ls_error_matrix,"+str(pxp.N)+".npy")

import pickle
config_dict = pickle.load(open("./data/"+str(pxp.N)+"/pxp,pxxxp,Hp_configs,"+str(N)+".obj","rb"))
keys = list(config_dict.keys())

coef_f0 = 0.12295995931667794
config_coef_diff = np.abs(config_coef_matrix - coef_f0 * np.ones((np.size(config_coef_matrix,axis=0),np.size(config_coef_matrix,axis=1))))
    
min_indices = np.where(np.abs(config_coef_diff) < 1e-3)
min_n = min_indices[0]
min_m = min_indices[1]
for n in range(0,np.size(min_n,axis=0)):
    print("\n")
    print(config_coef_diff[min_n[n],min_m[n]])
    print(model_space.basis[min_n[n]],model_space.basis[min_m[n]])

print("\n")
pxp_config = model_space.basis[min_n[0]]
pxxxp_config = model_space.basis[min_m[0]]
a = config_coef_matrix[min_n[0],min_m[0]]
# a=0
print(a)

Hp_pxp = np.zeros((pxp.dim,pxp.dim))
for n in range(0,pxp.dim):
    bits = pxp.basis[n]
    for m in range(0,pxp.N):
        if m == 0:
            mm1 = pxp.N-1
        else:
            mm1 = m -1
        if m == pxp.N-1:
            mp1 = 0
        else:
            mp1 = m +1
        if bits[mm1] == 0 and bits[mp1] == 0:
            pm = pxp_config[m]
            if pm == 1: #+
                if bits[m] == 0:
                    new_bits = np.copy(bits)
                    new_bits[m] = 1
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        Hp_pxp[pxp.keys[new_ref],n] += 1
            else:
                if bits[m] == 1:
                    new_bits = np.copy(bits)
                    new_bits[m] = 0
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        Hp_pxp[pxp.keys[new_ref],n] += 1

        
Hm_pxp = np.conj(np.transpose(Hp_pxp))

Hp_pxxxp = np.zeros((pxp.dim,pxp.dim))
for n in range(0,pxp.dim):
    bits = pxp.basis[n]
    for m in range(0,pxp.N):
        if m == 0:
            mm1 = pxp.N-1
            mm2 = pxp.N-2
        elif m == 1:
            mm1 = m -1
            mm2 = pxp.N-1
        else:
            mm1 = m - 1
            mm2 = m - 2

        if m == pxp.N-1:
            mp1 = 0
            mp2 = 1
        elif m == pxp.N-2:
            mp1 = m + 1
            mp2 = 0
        else:
            mp1 = m + 1
            mp2 = m + 2

        if bits[mm2] == 0 and bits[mp2] == 0:
            pm = pxxxp_config[m]
            if pm == 1: #-+-
                if bits[mm1] == 1 and bits[m] == 0 and bits[mp1] == 1:
                    new_bits = np.copy(bits)
                    new_bits[mm1] = 0
                    new_bits[m] = 1
                    new_bits[mp1] = 0
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        Hp_pxxxp[pxp.keys[new_ref],n] += 1
            else: #+-+
                if bits[mm1] == 0 and bits[m] == 1 and bits[mp1] == 0:
                    new_bits = np.copy(bits)
                    new_bits[mm1] = 1
                    new_bits[m] = 0
                    new_bits[mp1] = 1
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        Hp_pxxxp[pxp.keys[new_ref],n] += 1
                      
        
Hm_pxxxp = np.conj(np.transpose(Hp_pxxxp))

Hp = Hp_pxp + a * Hp_pxxxp

Hm = np.conj(np.transpose(Hp))

z=zm_state(2,1,pxp,1)

Hz = 1/2*com(Hp,Hm)
e,u = np.linalg.eigh(Hz)
temp = np.abs(np.vdot(u[:,0],z.prod_basis()))
print("Z2 Lw Overlap")
print(temp)
print("\n")

fsa_basis =z.prod_basis()
current_state = fsa_basis
fsa_dim = pxp.N
for n in range(0,fsa_dim):
    next_state = np.dot(Hp,current_state)
    if np.abs(np.vdot(next_state,next_state))>1e-5:
        next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
        fsa_basis = np.vstack((fsa_basis,next_state))
        current_state = next_state
    else:
        break
fsa_basis = np.transpose(fsa_basis)

Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
for n in range(0,np.size(Hz_exp,axis=0)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])

Hz_diff = np.zeros(np.size(Hz_exp)-1)
for n in range(0,np.size(Hz_diff,axis=0)):
    Hz_diff[n] = np.abs(Hz_exp[n+1] - Hz_exp[n])
    
#spacing error
error_matrix = np.zeros((np.size(Hz_diff),np.size(Hz_diff)))
for n in range(0,np.size(error_matrix,axis=0)):
    for m in range(0,np.size(error_matrix,axis=0)):
        error_matrix[n,m] = np.abs(Hz_diff[n] - Hz_diff[m])
error = np.power(np.trace(np.dot(error_matrix,np.conj(np.transpose(error_matrix)))),0.5)

for n in range(0,np.size(Hz_exp,axis=0)):
    print(Hz_exp[n],Hz_var[n])
print("\n")
for n in range(0,np.size(Hz_diff,axis=0)):
    print(Hz_diff[n])
print("\n")
    
print(error)
