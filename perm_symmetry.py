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
def is_zeros(H):
    temp = (np.abs(H)<1e-5).all()
    return temp

#init system
N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
t=np.arange(0,20,0.01)

H = spin_Hamiltonian(pxp,"x")
H.gen()
plt.matshow(np.abs(H.sector.matrix()))
plt.show()

uc_index = np.arange(0,int(N/2))
import itertools
perms = np.array(list(itertools.permutations(uc_index)))

def perm_matrix(N,perm):
    P = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis,axis=0)):
        bits = np.copy(pxp.basis[n])
        new_bits = np.copy(pxp.basis[n])
        for m in range(0,np.size(perm,axis=0)):
            loc = int(2*m)
            new_bits[loc] = bits[2*perm[m]]
            new_bits[loc+1] = bits[2*perm[m]+1]
        new_ref = bin_to_int_base_m(new_bits,pxp.base)
        if new_ref in pxp.basis_refs:
            new_index = pxp.keys[new_ref]
            P[new_index,n] +=1
    return P
P = dict()
pbar=ProgressBar()
for n in pbar(range(0,np.size(perms,axis=0))):
    P[n] = perm_matrix(N,perms[n])
# P_total = P[1]
# for n in range(2,len(P)):
    # P_total += P[n]
# plt.show()
# print(is_zeros(com(H.sector.matrix(),P_total)))

def frob_norm(A):
    return np.power(np.trace(np.dot(A,np.conj(np.transpose(A)))),0.5)

PCommutes = dict()
c=0
comNorm = np.zeros(len(P))
for n in range(0,len(P)):
    comNorm[n] = frob_norm(com(H.sector.matrix(),P[n]))
    if np.abs(comNorm[n])<1e-5:
        print(perms[n])
        PCommutes[c] = P[n]
        c+=1
P_total = PCommutes[1]
for n in range(2,len(PCommutes)):
    P_total += PCommutes[n]
plt.matshow(np.abs(P_total))
plt.show()
print(is_zeros(com(H.sector.matrix(),P_total)))
e,u = np.linalg.eigh(P_total)
print(e)
#form two perm sectors
permBasis1 = np.zeros(pxp.dim)
permBasis2 = np.zeros(pxp.dim)
for n in range(0,np.size(e,axis=0)):
    if np.abs(e[n]+1) <1e-5:
        permBasis1 = np.vstack((permBasis1,u[:,n]))
    else:
        permBasis2 = np.vstack((permBasis2,u[:,n]))
permBasis1 = np.transpose(np.delete(permBasis1,0,axis=0))
permBasis2 = np.transpose(np.delete(permBasis2,0,axis=0))

from Diagnostics import print_wf
for n in range(0,np.size(permBasis2,axis=1)):
    print("\n")
    print_wf(permBasis2[:,n],pxp,1e-2)

    
# new_basis = np.hstack((permBasis2,permBasis1))
# H_new = np.dot(np.conj(np.transpose(new_basis)),np.dot(H.sector.matrix(),new_basis))
# plt.matshow(np.abs(H_new))
# plt.show()

# z=zm_state(2,1,pxp)
# H.sector.find_eig()
# eig_overlap(z,H).plot()

# H_perm = np.dot(np.conj(np.transpose(permBasis2)),np.dot(H.sector.matrix(),permBasis2))
# psi_perm = np.dot(np.conj(np.transpose(permBasis2)),z.prod_basis())
# e,u = np.linalg.eigh(H_perm)
# overlap_perm = np.log10(np.abs(np.dot(np.conj(np.transpose(u)),psi_perm))**2)
# plt.scatter(e,overlap_perm,marker="x",color="red",s=100)
# plt.show()

# # z=zm_state(2,1,pxp)
# # overlap = eig_overlap(z,H).eval()
# # from Calculations import get_top_band_indices
# # scar_indices = get_top_band_indices(H.sector.eigvalues(),overlap,pxp.N,200,150,e_diff = 0.7)
# # plt.scatter(H.sector.eigvalues(),overlap)
# # for n in range(0,np.size(scar_indices,axis=0)):
    # # plt.scatter(H.sector.eigvalues()[scar_indices[n]],overlap[scar_indices[n]],marker="x",color="red",s=100)
# # plt.show()

# # scar_basis = H.sector.eigvectors()[:,scar_indices[0]]
# # for n in range(1,np.size(scar_indices,axis=0)):
    # # scar_basis = np.vstack((scar_basis,H.sector.eigvectors()[:,scar_indices[n]]))
# # scar_basis = np.transpose(scar_basis)

# # H_scar = np.dot(np.conj(np.transpose(scar_basis)),np.dot(H.sector.matrix(),scar_basis))
# # P_scar = np.dot(np.conj(np.transpose(scar_basis)),np.dot(P_total,scar_basis))

# # Hp = Hamiltonian(pxp)
# # Hp.site_ops[1] = np.array([[0,0],[1,0]])
# # Hp.site_ops[2] = np.array([[0,1],[0,0]])
# # Hp.model = np.array([[0,1,0],[0,2,0]])
# # Hp.model_coef = np.array([1,1])
# # Hp.uc_size = np.array([2,2])
# # Hp.uc_pos = np.array([1,0])
# # Hp.gen()
# # z=zm_state(2,1,pxp,1)
# # from Calculations import gen_fsa_basis
# # fsa_basis = gen_fsa_basis(Hp.sector.matrix(),z.prod_basis(),pxp.N)

# # H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# # P_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(P_total,fsa_basis))
# # plt.matshow(np.abs(com(H_fsa,P_fsa)))
# # plt.colorbar()
# # plt.show()
# # print(is_zeros(com(H_fsa,P_fsa)))
