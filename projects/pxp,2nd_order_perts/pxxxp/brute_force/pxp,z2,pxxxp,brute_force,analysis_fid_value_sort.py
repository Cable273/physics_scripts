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
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return np.real(exp(Q2,psi) - exp(Q,psi)**2)

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
    Hp = Hp_pxp + coef_f0 * Hp_pxxxp
    return Hp

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
# coef_f0 = 0
    
#find configs where Z2 ~ highest weight state
#No reflections in perturbed fidelity -> quenching from highest weight state
# indices = np.where(lw_overlap>=0.05)
# indices = np.where(lw_overlap>=-1)
indices = np.where(fsa_dim>=9)


to_del = []
#out of these configs, find the ones where the dim > N/2 (no trivials dim 2 subspaces)
for n in range(0,np.size(indices,axis=1)):
    n_index = indices[0][n]
    m_index = indices[1][n]

    if fsa_dim[n_index,m_index] < N:
        to_del = np.append(to_del,n)

index_var_error = np.zeros(np.size(indices,axis=1))
index_spacing_error = np.zeros(np.size(indices,axis=1))
for n in range(0,np.size(indices,axis=1)):
    n_index = indices[0][n]
    m_index = indices[1][n]

    index_var_error[n] = var_error[n_index,m_index]
    index_spacing_error[n] = spacing_error[n_index,m_index]
index_row_loc = np.arange(0,np.size(indices,axis=1))

#sort by spacing
# index_spacing_error,index_row_loc = (list(t) for t in zip(*sorted(zip(index_spacing_error,index_row_loc))))

#sort by var
index_var_error,index_row_loc = (list(t) for t in zip(*sorted(zip(index_var_error,index_row_loc))))

# for n in range(0,np.size(index_row_loc,axis=0)):
    # pxp_config = model_space.basis[indices[0][index_row_loc[n]]]
    # pxxxp_config = model_space.basis[indices[1][index_row_loc[n]]]
    # print(pxp_config,pxxxp_config)


count = 0
pxp_config = model_space.basis[indices[0][index_row_loc[count]]]
pxxxp_config = model_space.basis[indices[1][index_row_loc[count]]]


print(pxp_config,pxxxp_config)
# print(index_spacing_error[count])
# print(index_var_error[count])

Hp = gen_Hp(pxp_config,pxxxp_config)
Hm = np.conj(np.transpose(Hp))

Hp_test0 = Hamiltonian(pxp)
Hp_test0.site_ops[1] = np.array([[0,0],[1,0]])
Hp_test0.site_ops[2] = np.array([[0,1],[0,0]])
Hp_test0.model = np.array([[0,1,0],[0,2,0]])
Hp_test0.model_coef = np.array([1,1])
Hp_test0.uc_size = np.array([2,2])
Hp_test0.uc_pos = np.array([1,0])

Hp_test = Hamiltonian(pxp)
Hp_test.site_ops[1] = np.array([[0,0],[1,0]])
Hp_test.site_ops[2] = np.array([[0,1],[0,0]])
Hp_test.model = np.array([[0,2,1,2,0]])
Hp_test.model_coef = np.array([1])

Hp_test0.gen()
Hp_test.gen()
from Hamiltonian_Classes import H_operations
Hp_test = H_operations.add(Hp_test0,Hp_test,np.array([1,coef_f0]))
print((np.abs(Hp_test.sector.matrix()-Hp)<1e-5).all())

Hz = 1/2*com(Hp,Hm)

e,u = np.linalg.eigh(Hz)
z=zm_state(2,1,pxp,1)
lw = u[:,0]
print("\n|<lw | Z_2 > |")
print(np.abs(np.vdot(lw,z.prod_basis())))
print("\n")

from Calculations import gen_fsa_basis,gram_schmidt
fsa_basis = gen_fsa_basis(Hp,z.prod_basis(),pxp.N)
gs = gram_schmidt(fsa_basis)
gs.ortho()
fsa_basis = gs.ortho_basis

Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
Hz_var = np.zeros(np.size(fsa_basis,axis=1))
print("Exp/Var")
for n in range(0,np.size(Hz_exp,axis=0)):
    Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    Hz_var[n] = var(Hz,fsa_basis[:,n])
    print(Hz_exp[n],Hz_var[n])

print("\n Spacing")
Hz_diff = np.zeros(np.size(Hz_exp)-1)
for n in range(0,np.size(Hz_diff,axis=0)):
    Hz_diff[n] = Hz_exp[n+1] - Hz_exp[n]
    print(Hz_diff[n])

M = np.zeros((np.size(Hz_diff),np.size(Hz_diff)))
for n in range(0,np.size(M,axis=0)):
    for m in range(0,np.size(M,axis=0)):
        M[n,m] = np.abs(Hz_diff[n] - Hz_diff[m])
spacing_error = np.power(np.trace(np.dot(M,np.conj(np.transpose(M)))),0.5)
print("\nSpacing Error")
print(spacing_error)

print("\nMax Var")
print(np.max(Hz_var))

#check evolution in projected subspace
# H=Hp + Hm
# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H,fsa_basis))
# e,u = np.linalg.eigh(H_fsa)
# psi_energy = np.conj(u[0,:])
# t=np.arange(0,20,0.01)
# f=np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] =  np.abs(np.vdot(evolved_state,psi_energy))**2
# plt.plot(t,f)
# plt.show()
    
