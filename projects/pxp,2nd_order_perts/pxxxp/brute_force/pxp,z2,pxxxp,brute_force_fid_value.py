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

#pick pair of basis states (0100..) to build H+ so H+ + H- = PXP + a * PXXXP
model_space = unlocking_System([0,1],"open",2,N)
model_space.gen_basis()

a=0.122959959
#scan through all poss decompositions
spacing_error_matrix = np.zeros((model_space.dim,model_space.dim))
variance_error_matrix = np.zeros((model_space.dim,model_space.dim))
fsa_dim_matrix = np.zeros((model_space.dim,model_space.dim))
lw_overlap_matrix = np.zeros((model_space.dim,model_space.dim))
pbar=ProgressBar()
for config_i in pbar(range(0,model_space.dim)):
    for config_j in range(0,model_space.dim):
        pxp_config = model_space.basis[config_i]
        pxxxp_config = model_space.basis[config_j]

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

        Hz = 1/2*com(Hp,Hm)

        z=zm_state(2,1,pxp,1)
        from Calculations import gen_fsa_basis, gram_schmidt
        fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)
        if np.size(np.shape(fsa_basis))==2:
            gs = gram_schmidt(fsa_basis)
            gs.ortho()
            fsa_basis = gs.ortho_basis
            ez,uz = np.linalg.eigh(Hz)

            lw_overlap = np.abs(np.vdot(z.prod_basis(),uz[:,0]))
            fsa_dim = np.size(fsa_basis,axis=1)

            Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
            Hz_var = np.zeros(np.size(fsa_basis,axis=1))
            for n in range(0,np.size(fsa_basis,axis=1)):
                Hz_exp[n] = exp(Hz,fsa_basis[:,n])
                Hz_var[n] = var(Hz,fsa_basis[:,n])

            Hz_diff = np.zeros(np.size(Hz_exp)-1)
            for n in range(0,np.size(Hz_diff,axis=0)):
                # Hz_diff[n] = np.abs(Hz_exp[n+1] - Hz_exp[n])
                Hz_diff[n] = Hz_exp[n+1] - Hz_exp[n]

            M = np.zeros((np.size(Hz_diff),np.size(Hz_diff)))
            for n in range(0,np.size(M,axis=0)):
                for m in range(0,np.size(M,axis=0)):
                    M[n,m] = np.abs(Hz_diff[n] - Hz_diff[m])
            spacing_error = np.power(np.trace(np.dot(M,np.conj(np.transpose(M)))),0.5)
            variance_error = np.max(Hz_var)

            spacing_error_matrix[config_i,config_j] = spacing_error
            variance_error_matrix[config_i,config_j] = variance_error

            fsa_dim_matrix[config_i,config_j] = fsa_dim
            lw_overlap_matrix[config_i,config_j] = lw_overlap
        else:
            spacing_error_matrix[config_i,config_j] = 1000
            variance_error_matrix[config_i,config_j] = 1000
            fsa_dim_matrix[config_i,config_j] = 1
            lw_overlap_matrix[config_i,config_j] = 0

np.save("pxp,pxxxp,brute_force,f0_coef,spacing_error_matrix,"+str(pxp.N),spacing_error_matrix)
np.save("pxp,pxxxp,brute_force,f0_coef,variance_error_matrix,"+str(pxp.N),variance_error_matrix)
np.save("pxp,pxxxp,brute_force,f0_coef,fsa_dim_matrix,"+str(pxp.N),fsa_dim_matrix)
np.save("pxp,pxxxp,brute_force,f0_coef,lw_overlap_matrix,"+str(pxp.N),lw_overlap_matrix)
