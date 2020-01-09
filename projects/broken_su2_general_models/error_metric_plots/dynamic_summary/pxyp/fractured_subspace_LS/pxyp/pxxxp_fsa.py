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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,translational_general,PT,inversion
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
N=12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

pxp_syms = model_sym_data(pxp,[translational_general(pxp,order=4),PT(pxp)])

Hp = Hamiltonian(pxp,pxp_syms)
Hp.site_ops[1] = np.array([[0,0],[1,0]])
Hp.site_ops[2] = np.array([[0,1],[0,0]])
Hp.model = np.array([[0,1,2,0],[0,2,1,0]])
Hp.model_coef = np.array([1,1])
Hp.uc_size = np.array([2,2])
Hp.uc_pos = np.array([1,0])

# Hp = Hamiltonian(pxp,pxp_syms)
# Hp.site_ops[1] = np.array([[0,0],[1,0]])
# Hp.site_ops[2] = np.array([[0,1],[0,0]])
# Hp.site_ops[4] = np.array([[0,0],[0,1]])
# Hp.model = np.array([[0,1,2,1,0,4,0],[0,1,2,1,0,4,0],[0,4,0,1,2,1,0],[0,4,0,1,2,1,0]])
# Hp.model_coef = np.array([1,1,1,1])
# Hp.uc_size = np.array([4,4,4,4])
# Hp.uc_pos = np.array([1,3,3,1])

z=zm_state(4,1,pxp)
k=pxp_syms.find_k_ref(z.ref)
print(k)
# for n in range(0,np.size(k,axis=0)):
    # Hp.gen(k[n])
Hp.gen()
Hm = Hp.herm_conj()
# Hz = 1/2 * com(Hp.sector.matrix(k[0]),Hm.sector.matrix(k[0]))
Hz = 1/2 * com(Hp.sector.matrix(),Hm.sector.matrix())
plt.matshow(np.abs(Hz))
plt.show()
e,u = np.linalg.eigh(Hz)
print(e)
from Diagnostics import print_wf
print_wf(u[:,0],pxp,1e-2)
print("\n")
print_wf(u[:,1],pxp,1e-2)
print("\n")
print_wf(u[:,2],pxp,1e-2)
print("\n")
print_wf(u[:,3],pxp,1e-2)
print("\n")
print_wf(u[:,4],pxp,1e-2)
print("\n")
print_wf(u[:,5],pxp,1e-2)
print("\n")
print_wf(u[:,6],pxp,1e-2)
print("\n")
print_wf(u[:,7],pxp,1e-2)

# from Calculations import gen_fsa_basis
# fsa_basis = gen_fsa_basis(Hp.sector.matrix(),z.prod_basis(),int(2*pxp.N/4))
# H = H_operations.add(Hp,Hm,np.array([1,1]))
# H.sector.find_eig()
# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# e,u = np.linalg.eigh(H_fsa)
# overlap_fsa = np.log10(np.abs(u[0,:])**2)

# print(H.sector.matrix())
# eig_overlap(z,H).plot(tol=-15)
# plt.scatter(e,overlap_fsa,marker="x",color="red",s=100)
# plt.show()
# fidelity(z,H).plot(np.arange(0,20,0.01),z)
# plt.show()
# H = H_operations.add(Hp,Hm,np.array([1,1]))
# H.sector.find_eig(k[0])

# eig_overlap(z,H,k[0]).plot(tol=-15)
# plt.show()
# fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
# plt.show()

# # from Calculations import gen_fsa_basis
# # fsa_basis = gen_fsa_basis(Hp.sector.matrix(k[0]),z.sym_basis(k[0],pxp_syms),int(2*pxp.N/4))
# # H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(k[0]),fsa_basis))
# # e,u = np.linalg.eigh(H_fsa)
# # overlap_fsa = np.log10(np.abs(u[0,:])**2)


# # u_comp = np.dot(fsa_basis,u)
# # exact_overlap = np.zeros(np.size(u_comp,axis=1))
# # for n in range(0,np.size(exact_overlap,axis=0)):
    # # max_overlap = 0
    # # for m in range(0,np.size(H.sector.eigvectors(k[0]),axis=1)):
        # # temp = np.abs(np.vdot(H.sector.eigvectors(k[0])[:,m],u_comp[:,n]))**2
        # # if temp > max_overlap:
            # # max_overlap = temp
    # # exact_overlap[n] = max_overlap
# # plt.plot(e,exact_overlap,marker="s")
# # plt.show()
        
