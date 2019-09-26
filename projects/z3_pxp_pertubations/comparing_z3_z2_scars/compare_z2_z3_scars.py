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

N = 18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)

z2=zm_state(2,1,pxp)
z3=zm_state(3,1,pxp)

#sym sectors
k2 = pxp_syms.find_k_ref(z2.ref)
k3 = pxp_syms.find_k_ref(z3.ref)
all_k = k2

def sym_key(k):
    return bin_to_int_base_m(k,pxp.N+1)

for n in range(0,np.size(all_k,axis=0)):
    H.gen(all_k[n])
    H.sector.find_eig(all_k[n])

U = dict()
c=0
for n in range(0,np.size(all_k,axis=0)):
    U[sym_key(all_k[n])] = pxp_syms.basis_transformation(all_k[n])
    c = c + np.shape(U[sym_key(all_k[n])])[1]
    print(np.shape(U[sym_key(all_k[n])]))

# rotate to full basis and direct sum sym eigenstates
eigvectors_comp = dict()
for n in range(0,np.size(all_k,axis=0)):
    eigvectors_comp[sym_key(all_k[n])] = np.dot(U[sym_key(all_k[n])],H.sector.eigvectors(all_k[n]))

eigvectors_comp_comb = eigvectors_comp[sym_key(all_k[0])]
eigvalues_comb = H.sector.eigvalues(all_k[0])
for n in range(1,np.size(all_k,axis=0)):
    eigvectors_comp_comb = np.hstack((eigvectors_comp_comb,eigvectors_comp[sym_key(all_k[n])]))
    eigvalues_comb = np.append(eigvalues_comb,H.sector.eigvalues(all_k[n]))

eigvalues_no_parity_sectors = np.load("./z3,no_parity,eigvalues,"+str(pxp.N)+".npy")
eigvectors_no_parity_sectors = np.load("./z3,no_parity,eigvectors,"+str(pxp.N)+".npy")

eigvalues_comb = np.append(eigvalues_comb,eigvalues_no_parity_sectors)
eigvectors_comp_comb = np.hstack((eigvectors_comp_comb,eigvectors_no_parity_sectors))
np.save("pxp,no_pert,e,"+str(pxp.N),eigvalues_comb)
np.save("pxp,no_pert,u,"+str(pxp.N),eigvectors_comp_comb)

# z2_overlap = np.log10(np.abs(eigvectors_comp_comb[pxp.keys[z2.ref],:])**2)
# z3_overlap = np.log10(np.abs(eigvectors_comp_comb[pxp.keys[z3.ref],:])**2)

# # top band (scar states)
# from Calculations import get_top_band_indices
# z2_scar_indices = get_top_band_indices(eigvalues_comb,z2_overlap,N,110,400,e_diff = 1)
# z3_scar_indices = get_top_band_indices(eigvalues_comb,z3_overlap,int(2*N/3),110,400,e_diff = 1)
# print(np.size(z2_scar_indices))
# print(np.size(z3_scar_indices))
# print(z2_scar_indices)
# print(z3_scar_indices)
# # #check scars identified properly

# plt.scatter(eigvalues_comb,z2_overlap)
# for n in range(0,np.size(z2_scar_indices,axis=0)):
    # plt.scatter(eigvalues_comb[z2_scar_indices[n]],z2_overlap[z2_scar_indices[n]],color="red",marker="x")
# plt.show()
# plt.scatter(eigvalues_comb,z3_overlap)
# for n in range(0,np.size(z3_scar_indices,axis=0)):
    # plt.scatter(eigvalues_comb[z3_scar_indices[n]],z3_overlap[z3_scar_indices[n]],color="red",marker="x")
    # print(eigvalues_comb[z3_scar_indices[n]],z3_overlap[z3_scar_indices[n]])
# plt.show()

# #scars in full comp basis
# z2_scars = np.zeros(pxp.dim)
# for n in range(0,np.size(z2_scar_indices,axis=0)):
    # z2_scars = np.vstack((z2_scars,eigvectors_comp_comb[:,z2_scar_indices[n]]))
# z2_scars = np.transpose(np.delete(z2_scars,0,axis=0))

# z3_scars = np.zeros(pxp.dim)
# for n in range(0,np.size(z3_scar_indices,axis=0)):
    # z3_scars = np.vstack((z3_scars,eigvectors_comp_comb[:,z3_scar_indices[n]]))
# z3_scars = np.transpose(np.delete(z3_scars,0,axis=0))

# print("\n")
# print("DIFF SCARS")
# tol = 8

# eigenvalues = np.copy(eigvalues_comb)
# overlap = np.copy(z3_overlap)
# to_del=[]
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-tol:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
# plt.scatter(eigenvalues,overlap)
# # plt.scatter(eigvalues_comb,z3_overlap)

# other_scar_e = np.zeros(np.size(z2_scar_indices))
# other_scar_overlap = np.zeros(np.size(z2_scar_indices))
# for n in range(0,np.size(z2_scar_indices,axis=0)):
    # other_scar_e[n] = eigvalues_comb[z2_scar_indices[n]]
    # other_scar_overlap[n] = z3_overlap[z2_scar_indices[n]]
# plt.scatter(other_scar_e,other_scar_overlap,marker="D",color="red",alpha=0.6,s=100,label=r"$Z_2$ Scars")
# plt.legend()
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\vert \langle Z_3 \vert E \rangle \vert^2$")
# plt.title(r"$PXP$ N="+str(pxp.N))
# plt.legend()
# plt.show()

# eigenvalues = np.copy(eigvalues_comb)
# overlap = np.copy(z2_overlap)
# to_del=[]
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-tol:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
# print("\n")
# plt.scatter(eigenvalues,overlap)
# # plt.scatter(eigvalues_comb,z2_overlap)

# other_scar_e = np.zeros(np.size(z2_scar_indices))
# other_scar_overlap = np.zeros(np.size(z2_scar_indices))
# for n in range(0,np.size(z3_scar_indices,axis=0)):
    # other_scar_e[n] = eigvalues_comb[z3_scar_indices[n]]
    # other_scar_overlap[n] = z2_overlap[z3_scar_indices[n]]
# plt.scatter(other_scar_e,other_scar_overlap,marker="D",color="red",alpha=0.6,s=100,label=r"$Z_3$ Scars")
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\vert \langle Z_2 \vert E \rangle \vert^2$")
# plt.legend()
# plt.title(r"$PXP$ N="+str(pxp.N))
# plt.show()

# M_overlap = np.zeros((np.size(z2_scar_indices),np.size(z3_scar_indices)),dtype=complex)
# for n in range(0,np.size(z2_scar_indices,axis=0)):
    # for m in range(0,np.size(z3_scar_indices,axis=0)):
        # M_overlap[n,m] = np.vdot(eigvectors_comp_comb[:,z2_scar_indices[n]],eigvectors_comp_comb[:,z3_scar_indices[m]])
# print(M_overlap)
# plt.matshow(np.abs(M_overlap))
# plt.xlabel
# plt.title(r"$M_{nm} = \vert \langle (Z_2^{scar})_n \vert (Z_3^{scar})_m \rangle \vert, PXP, N=$"+str(pxp.N))
# plt.xlabel(r"$m$")
# plt.ylabel(r"$n$")
# plt.colorbar()
# plt.show()
    
