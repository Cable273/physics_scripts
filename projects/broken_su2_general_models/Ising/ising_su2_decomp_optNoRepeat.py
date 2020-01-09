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
    return np.real(np.vdot(psi,np.dot(Q,psi)))
def var(Q,psi):
    Q2 = np.dot(Q,Q)
    return exp(Q2,psi)-exp(Q,psi)**2

N=12
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()
# pxp = pxp.U1_sector(5)
print(pxp.dim)
pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=2)])

# J = -1
# hx = 0.05
# hz = -0.5

J = 1
hx = 1
hz = 1
Hp = dict()
Hp[0] = Hamiltonian(pxp,pxp_syms)
Hp[0].site_ops[1] = np.array([[0,0],[1,0]])
Hp[0].site_ops[2] = np.array([[0,1],[0,0]])
Hp[0].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[0].model = np.array([[1,1],[2,2],[1,2],[2,1],[1],[2],[3]])
Hp[0].model_coef = np.array([J/4,J/4,J/4,J/4,hx/2,hx/2,hz/2])
Hp[0].uc_size = np.array([2,2,2,2,2,2,1])
Hp[0].uc_pos = np.array([0,1,0,1,0,1,0])

Hp[1] = Hamiltonian(pxp,pxp_syms)
Hp[1].site_ops[1] = np.array([[0,0],[1,0]])
Hp[1].site_ops[2] = np.array([[0,1],[0,0]])
Hp[1].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[1].site_ops[4] = np.array([[0,0],[0,1]])
Hp[1].model = np.array([[1,3,3,2],[1,3,3,1],[1,3,3,1],[2,3,3,1],[2,3,3,2],[2,3,3,2],[2,3,3,1],[1,3,3,2]])
Hp[1].model_coef = np.array([1,1,1,1,1,1,1,1])
Hp[1].uc_size = np.array([2,2,2,2,2,2,2,2])
Hp[1].uc_pos = np.array([1,0,1,0,0,1,1,0])

Hp[2] = Hamiltonian(pxp,pxp_syms)
Hp[2].site_ops[1] = np.array([[0,0],[1,0]])
Hp[2].site_ops[2] = np.array([[0,1],[0,0]])
Hp[2].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[2].site_ops[4] = np.array([[0,0],[0,1]])
Hp[2].model = np.array([[1,1,1],[2,2,2],[1,1,2],[2,2,1]])
Hp[2].model_coef = np.array([1,1,1,1])
Hp[2].uc_size = np.array([2,2,2,2])
Hp[2].uc_pos = np.array([1,0,1,0])

Hp[3] = Hamiltonian(pxp,pxp_syms)
Hp[3].site_ops[1] = np.array([[0,0],[1,0]])
Hp[3].site_ops[2] = np.array([[0,1],[0,0]])
Hp[3].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[3].site_ops[4] = np.array([[0,0],[0,1]])
Hp[3].model = np.array([[1,3,3],[2,3,3]])
Hp[3].model_coef = np.array([1,1])

Hp[4] = Hamiltonian(pxp,pxp_syms)
Hp[4].site_ops[1] = np.array([[0,0],[1,0]])
Hp[4].site_ops[2] = np.array([[0,1],[0,0]])
Hp[4].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[4].site_ops[4] = np.array([[0,0],[0,1]])
Hp[4].model = np.array([[3,3,1],[3,3,2]])
Hp[4].model_coef = np.array([1,1])

Hp[5] = Hamiltonian(pxp,pxp_syms)
Hp[5].site_ops[1] = np.array([[0,0],[1,0]])
Hp[5].site_ops[2] = np.array([[0,1],[0,0]])
Hp[5].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[5].site_ops[4] = np.array([[0,0],[0,1]])
Hp[5].model = np.array([[1,3,1],[2,3,2]])
Hp[5].model_coef = np.array([1,1])
Hp[5].uc_size = np.array([2,2])
Hp[5].uc_pos = np.array([1,0])

Hp[6] = Hamiltonian(pxp,pxp_syms)
Hp[6].site_ops[1] = np.array([[0,0],[1,0]])
Hp[6].site_ops[2] = np.array([[0,1],[0,0]])
Hp[6].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[6].site_ops[4] = np.array([[0,0],[0,1]])
Hp[6].model = np.array([[2,2,1],[1,1,2],[2,2,2],[1,1,1]])
Hp[6].model_coef = np.array([1,1,1,1])
Hp[6].uc_size = np.array([2,2,2,2])
Hp[6].uc_pos = np.array([1,0,1,0])

Hp[7] = Hamiltonian(pxp,pxp_syms)
Hp[7].site_ops[1] = np.array([[0,0],[1,0]])
Hp[7].site_ops[2] = np.array([[0,1],[0,0]])
Hp[7].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[7].site_ops[4] = np.array([[0,0],[0,1]])
Hp[7].model = np.array([[2,1,1],[1,2,2],[1,2,1],[2,1,2]])
Hp[7].model_coef = np.array([1,1,1,1])
Hp[7].uc_size = np.array([2,2,2,2])
Hp[7].uc_pos = np.array([1,0,0,1])

Hp[8] = Hamiltonian(pxp,pxp_syms)
Hp[8].site_ops[1] = np.array([[0,0],[1,0]])
Hp[8].site_ops[2] = np.array([[0,1],[0,0]])
Hp[8].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[8].site_ops[4] = np.array([[0,0],[0,1]])
Hp[8].model = np.array([[3,1],[1,3],[3,2],[2,3]])
Hp[8].model_coef = np.array([1,1,1,1])
Hp[8].uc_size = np.array([2,2,2,2])
Hp[8].uc_pos = np.array([0,0,1,1])

Hp[9] = Hamiltonian(pxp,pxp_syms)
Hp[9].site_ops[1] = np.array([[0,0],[1,0]])
Hp[9].site_ops[2] = np.array([[0,1],[0,0]])
Hp[9].site_ops[3] = np.array([[-1/2,0],[0,1/2]])
Hp[9].site_ops[4] = np.array([[0,0],[0,1]])
Hp[9].model = np.array([[2,3,1],[1,3,2]])
Hp[9].model_coef = np.array([1,1])
Hp[9].uc_size = np.array([2,2])
Hp[9].uc_pos = np.array([1,0])

k=[0]
for n in range(0,len(Hp)):
    print(n)
    Hp[n].gen(k)

def fidelity_eval(psi_energy,e,t):
    evolved_state = time_evolve_state(psi_energy,e,t)
    f = np.abs(np.vdot(evolved_state,psi_energy))**2
    return -f

from copy import deepcopy
from Hamiltonian_Classes import H_operations
from scipy.optimize import minimize,minimize_scalar
def fidelity_error(coef):
    np.save("NEWising,pert_coef,"+str(pxp.N),coef)
    Hp_total = deepcopy(Hp[0])
    for n in range(1,len(Hp)):
        Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
    Hm = np.conj(np.transpose(Hp_total.sector.matrix()))
    Hz = 1/2 * com(Hp_total.sector.matrix(),Hm)
    e,u = np.linalg.eigh(Hz)
    psi = u[:,0]

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

# coef = np.load("./ising,pert_coef,8.npy")
coef = np.load("./NEWising,pert_coef,10.npy")
# print(coef)
# coef = np.zeros(9)
# from scipy.optimize import minimize
# res = minimize(lambda coef: fidelity_error(coef),method="Nelder-Mead",x0=coef)
# res = minimize(lambda coef: fidelity_error(coef),method="SLSQP",x0=coef)
# # # # # # res = minimize(lambda coef: spacing_error(coef,psi),method="Nelder-Mead",x0=coef)

# coef = res.x
# # np.save("couplings2ising,pert_coef,"+str(pxp.N),coef)
# np.save("NEWising,pert_coef,"+str(pxp.N),coef)
# coef = 0

Hp_total = deepcopy(Hp[0])
for n in range(1,len(Hp)):
    Hp_total = H_operations.add(Hp_total,Hp[n],np.array([1,coef[n-1]]))
Hm = Hp_total.herm_conj()

H = H_operations.add(Hp_total,Hm,np.array([1,1]))
H.sector.find_eig(k)
ls = level_stats(H.sector.eigvalues(k))
print(H.sector.eigvalues(k))
print(ls.mean())
# Hz = 1/2 * com(Hp_total.sector.matrix(),Hm.sector.matrix())
# e,u = np.linalg.eigh(Hz)
# psi = u[:,0]
    
# # np.save("xy,pert,lw_state,"+str(pxp.N),psi)

# from Calculations import gen_fsa_basis,gram_schmidt
# fsa_basis = gen_fsa_basis(Hp_total.sector.matrix(),psi,pxp.N)
# gs = gram_schmidt(fsa_basis)
# gs.ortho()
# fsa_basis = gs.ortho_basis

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

# H = H_operations.add(Hp_total,Hm,np.array([1,1]))
# H.sector.find_eig()

# e = H.sector.eigvalues()
# u = H.sector.eigvectors()

# psi_energy = np.dot(np.conj(np.transpose(u)),psi)
# eigenvalues = e
# overlap = np.log10(np.abs(psi_energy)**2)
# to_del=[]
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-10:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
# plt.scatter(eigenvalues,overlap)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# # plt.title(r"Ising + SU(2) perts, SU(2) Lowest weight overlap, $N=$"+str(pxp.N))
# plt.title(r"Ising, SU(2) Lowest weight overlap, $N=$"+str(pxp.N))
# plt.show()

# t=np.arange(0,20,0.01)
# f=np.zeros(np.size(t))
# pol = ref_state(0,pxp)
# pol_energy = np.conj(u[0,:])
# f_pol = np.zeros(np.size(t))
# f_hw = np.zeros(np.size(t))
# psi_HW_energy = np.dot(np.conj(np.transpose(H.sector.eigvectors())),fsa_basis[:,np.size(fsa_basis,axis=1)-1])
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] = np.abs(np.vdot(psi_energy,evolved_state))**2
    # f_pol[n] = np.abs(np.vdot(pol_energy,evolved_state))**2
    # f_hw[n] = np.abs(np.vdot(psi_HW_energy,evolved_state))**2
# plt.plot(t,f,label=r"$\vert H_z, LW \rangle$")
# plt.plot(t,f_pol,label=r"$\vert 000...\rangle$")
# plt.plot(t,f_hw,label=r"$\vert H_z, HW \rangle$")
# plt.legend()
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# # plt.title(r"Ising, SU(2) Lowest weight quench, $N=$"+str(pxp.N))
# # plt.title(r"Ising + SU(2) perts, SU(2) Lowest weight quench, $N=$"+str(pxp.N))
# plt.title(r"Ising, SU(2) Lowest weight quench, $N=$"+str(pxp.N))
# plt.show()

# pbar=ProgressBar()
# for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    # z=ref_state(pxp.basis_refs[n],pxp)
    # fidelity(z,H).plot(np.arange(0,20,0.1),z)
# plt.show()

# # entropy
# # ent = entropy(pxp)
# # ent_vals = np.zeros(np.size(e))
# # pbar=ProgressBar()
# # for n in pbar(range(0,np.size(ent_vals,axis=0))):
    # # ent_vals[n] = ent.eval(u[:,n])
# # plt.scatter(e,ent_vals)
# # plt.xlabel(r"$E$")
# # plt.xlabel(r"$S$")
# # plt.title(r"$H=P(XX+YY)P$ Perturbed (1st Order), Entropy, $N=$"+str(pxp.N))
# # plt.show()

# # np.savetxt("xy,Hz,exp,"+str(pxp.N),Hz_exp)
# # np.savetxt("xy,Hz,var,"+str(pxp.N),Hz_var)
# # np.savetxt("xy,Hz,diff,"+str(pxp.N),Hz_diff)
