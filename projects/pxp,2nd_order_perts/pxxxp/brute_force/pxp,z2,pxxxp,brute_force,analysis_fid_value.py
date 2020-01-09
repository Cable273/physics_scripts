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

def gen_Hp(pxp_config,pxxxp_config):
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
    return Hp_pxp,Hp_pxxxp

N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

model_space = unlocking_System([0,1],"open",2,N)
model_space.gen_basis()

spacing_error = np.load("./f0_coef/"+str(N)+"/pxp,pxxxp,brute_force,f0_coef,spacing_error_matrix,"+str(N)+".npy")
var_error = np.load("./f0_coef/"+str(N)+"/pxp,pxxxp,brute_force,f0_coef,variance_error_matrix,"+str(N)+".npy")
fsa_dim  = np.load("./f0_coef/"+str(N)+"/pxp,pxxxp,brute_force,f0_coef,fsa_dim_matrix,"+str(N)+".npy")
lw_overlap  = np.load("./f0_coef/"+str(N)+"/pxp,pxxxp,brute_force,f0_coef,lw_overlap_matrix,"+str(N)+".npy")

coef_f0 = 0.12295995931667794
    
spacing_indices = np.where(np.abs(spacing_error)<1e-1)
print(np.shape(spacing_indices))
var_indices = np.where(np.abs(var_error)<1e-2)

spacing_n = spacing_indices[0]
spacing_m = spacing_indices[1]

var_n = var_indices[0]
var_m = var_indices[1]

space_error_fsa_dim = np.zeros(np.size(spacing_n))
var_error_fsa_dim = np.zeros(np.size(var_n))

pbar=ProgressBar()
# for n in pbar(range(0,np.size(space_error_fsa_dim,axis=0))):
    # pxp_config = model_space.basis[spacing_n[n]]
    # pxxxp_config = model_space.basis[spacing_m[n]]

    # Hp_pxp, Hp_pxxxp = gen_Hp(pxp_config,pxxxp_config)
    # a=coef_f0
    # Hp = Hp_pxp + a * Hp_pxxxp

    # z=zm_state(2,1,pxp,1)
    # from Calculations import gen_fsa_basis
    # fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)
    # space_error_fsa_dim[n] = np.size(fsa_basis,axis=1)

pbar=ProgressBar()
for n in pbar(range(0,np.size(var_error_fsa_dim,axis=0))):
    pxp_config = model_space.basis[var_n[n]]
    pxxxp_config = model_space.basis[var_m[n]]

    Hp_pxp, Hp_pxxxp = gen_Hp(pxp_config,pxxxp_config)
    a=coef_f0
    Hp = Hp_pxp + a * Hp_pxxxp

    z=zm_state(2,1,pxp,1)
    from Calculations import gen_fsa_basis
    fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)
    var_error_fsa_dim[n] = np.size(fsa_basis,axis=1)
# print(space_error_fsa_dim)
# print(np.max(space_error_fsa_dim))
print(var_error_fsa_dim)
c=[]
for n in range(0,np.size(var_error_fsa_dim,axis=0)):
    if var_error_fsa_dim[n] == pxp.N+1:
        c = np.append(c,n)

for count in range(0,np.size(c,axis=0)):
    a = int(c[count])
    pxp_config = model_space.basis[var_n[a]]
    pxxxp_config = model_space.basis[var_m[a]]
    Hp_pxp,Hp_pxxxp = gen_Hp(pxp_config,pxxxp_config)

    Hp = Hp_pxp + coef_f0 * Hp_pxxxp
    Hm = np.conj(np.transpose(Hp))
    Hz = 1/2 * com(Hp,Hm)

    z=zm_state(2,1,pxp,1)
    fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)

    e,u = np.linalg.eigh(Hz)
    lw = u[:,0]
    overlap = np.abs(np.vdot(lw,z.prod_basis()))
    if overlap > 0.01:
        print("\n")
        print(pxp_config,pxxxp_config)
        from Diagnostics import print_wf
        print("\n|<LW Hz | Z2 >|")
        # print_wf(lw,pxp,1e-2)
        print("Z2 overlap")
        print(overlap)

        Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
        Hz_var = np.zeros(np.size(fsa_basis,axis=1))
        print("\nExp/Var")
        for n in range(0,np.size(Hz_exp,axis=0)):
            Hz_exp[n] = exp(Hz,fsa_basis[:,n])
            Hz_var[n] = var(Hz,fsa_basis[:,n])
            print(Hz_exp[n],Hz_var[n])

        print("\nHz Level spacing")
        Hz_diff = np.zeros(np.size(Hz_exp)-1)
        for n in range(0,np.size(Hz_diff,axis=0)):
            Hz_diff[n] = Hz_exp[n+1] - Hz_exp[n]
            print(Hz_diff[n])
