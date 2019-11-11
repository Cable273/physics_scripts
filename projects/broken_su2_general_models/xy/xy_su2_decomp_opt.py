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
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

N=14
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

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

for n in range(0,len(Hp)):
    Hp[n].gen()

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

from copy import deepcopy
from Hamiltonian_Classes import H_operations
from scipy.optimize import minimize,minimize_scalar
def fidelity_error(coef,psi):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    H=Hp_total.sector.matrix()+Hm
    e,u = np.linalg.eigh(H)
    psi_energy = np.dot(np.conj(np.transpose(u)),psi)

    t=np.arange(0,10,0.01)
    f=np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        f[n] = -fidelity_eval(psi_energy,e,t[n])
    for n in range(0,np.size(f,axis=0)):
        if f[n] < 0.1:
            cut = n
            break
    f_max = np.max(f[cut:])
        
    res = minimize_scalar(lambda t: fidelity_eval(psi_energy,e,t),method="golden",bracket=(4.5,5.5))
    f = -fidelity_eval(psi_energy,e,res.x)
    print(coef,f)
    # print(f)
    if res.x <1e-5:
        return 1000
    else:
        return -f_max

def spacing_error(coef,psi):
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))

    Hz = 1/2 * com(Hp_total.sector.matrix(),Hm)

    from Calculations import gen_fsa_basis
    fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),psi,pxp.N)

    Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
    for n in range(0,np.size(fsa_basis,axis=1)):
        Hz_exp[n] = exp(Hz,fsa_basis[:,n])

    Hz_diff = np.zeros(np.size(Hz_exp)-1)
    for n in range(0,np.size(Hz_diff,axis=0)):
        Hz_diff[n] = Hz_exp[n+1] - Hz_exp[n]
    M = np.zeros((np.size(Hz_diff),np.size(Hz_diff)))
    for n in range(0,np.size(M,axis=0)):
        for m in range(0,np.size(M,axis=0)):
            M[n,m] = np.abs(Hz_diff[n] - Hz_diff[m])
    error = np.power(np.trace(np.dot(M,np.conj(np.transpose(M)))),0.5)
    print(coef,error)
    return error
    
coef = np.zeros(2)
Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    
Hp0 = Hp_total.sector.matrix()
Hm0 = np.conj(np.transpose(Hp0))
Hz0 = 1/2 * com(Hp0,Hm0)
# psi = bin_state(np.array([0,1,0,1,1,0,0,1,0,1,1,0]),pxp).prod_basis()
# print(exp(Hz0,psi))
# print(var(Hz0,psi))

ez,uz = np.linalg.eigh(Hz0)
print(np.unique(ez))
from Diagnostics import print_wf
count = 0
psi = uz[:,count]
from Diagnostics import print_wf
print("\nLw")
print_wf(psi,pxp,1e-2)
print("\n")

# coef = np.load("./data/xy,pert_coef,8.npy")
# # coef = np.zeros(2)
# # from scipy.optimize import minimize
# # # res = minimize(lambda coef: fidelity_error(coef,psi),method="Nelder-Mead",x0=coef)
# # res = minimize(lambda coef: spacing_error(coef,psi),method="Nelder-Mead",x0=coef)

# # coef = res.x
# # np.save("xy,pert_coef,"+str(pxp.N),coef)

# Hp_total = deepcopy(Hp[0])
# for n in range(1,len(Hp)):
    # Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    
# Hp = Hp_total.sector.matrix()
# Hm = np.conj(np.transpose(Hp))
# Hz = 1/2 * com(Hp,Hm)
# ez,uz = np.linalg.eigh(Hz)
# print(ez[count])
# psi = uz[:,count]
# np.save("xy,pert,lw_state,"+str(pxp.N),psi)

# from Calculations import gen_fsa_basis
# fsa_basis = gen_fsa_basis(Hp,psi,pxp.N)

# Hz_exp = np.zeros(np.size(fsa_basis,axis=1))
# Hz_var = np.zeros(np.size(fsa_basis,axis=1))
# print("\nExp/Var")
# for n in range(0,np.size(fsa_basis,axis=1)):
    # Hz_exp[n] = exp(Hz,fsa_basis[:,n])
    # Hz_var[n] = var(Hz,fsa_basis[:,n])
    # print(Hz_exp[n],Hz_var[n])

# print("\nHz Spacing")
# Hz_diff = np.zeros(np.size(Hz_exp)-1)
# for n in range(0,np.size(Hz_diff,axis=0)):
    # Hz_diff[n] = Hz_exp[n+1] - Hz_exp[n]
    # print(Hz_diff[n])

# H = Hp + Hm
# plt.matshow(np.abs(H))
# plt.show()
# e,u = np.linalg.eigh(H)

# psi_energy = np.dot(np.conj(np.transpose(u)),psi)
# t=np.arange(0,40,0.01)
# f=np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] = np.abs(np.vdot(psi_energy,evolved_state))**2
# plt.plot(t,f)
# plt.show()

# #entropy
# ent = entropy(pxp)
# ent_vals = np.zeros(np.size(e))
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(ent_vals,axis=0))):
    # ent_vals[n] = ent.eval(u[:,n])
# plt.scatter(e,ent_vals)
# plt.show()
    
