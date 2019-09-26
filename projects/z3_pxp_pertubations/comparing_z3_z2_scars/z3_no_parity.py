#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sparse_linalg
import sys
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/Classes/'
sys.path.append(file_dir)
file_dir = '/localhome/pykb/physics_code/Exact_Diagonalization/functions/'
sys.path.append(file_dir)

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
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

N = 21
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H = spin_Hamiltonian(pxp,"x",pxp_syms)

z2=zm_state(2,1,pxp)
z3=zm_state(3,1,pxp)

#sym sectors
k2 = pxp_syms.find_k_ref(z2.ref)
k3 = pxp_syms.find_k_ref(z3.ref)
k3 = np.delete(k3,0,axis=0)
all_k = k3

def sym_key(k):
    return bin_to_int_base_m(k,pxp.N+1)

for n in range(0,np.size(all_k,axis=0)):
    H.gen(all_k[n])
    H.sector.find_eig(all_k[n])

U = dict()
for n in range(0,np.size(all_k,axis=0)):
    U[sym_key(all_k[n])] = pxp_syms.basis_transformation(all_k[n])

#rotate to full basis and direct sum sym eigenstates
eigvectors_comp = dict()
for n in range(0,np.size(all_k,axis=0)):
    eigvectors_comp[sym_key(all_k[n])] = np.dot(U[sym_key(all_k[n])],H.sector.eigvectors(all_k[n]))

eigvectors_comp_comb = eigvectors_comp[sym_key(all_k[0])]
eigvalues_comb = H.sector.eigvalues(all_k[0])
for n in range(1,np.size(all_k,axis=0)):
    eigvectors_comp_comb = np.hstack((eigvectors_comp_comb,eigvectors_comp[sym_key(all_k[n])]))
    eigvalues_comb = np.append(eigvalues_comb,H.sector.eigvalues(all_k[n]))
np.save("z3,no_parity,eigvalues,"+str(pxp.N),eigvalues_comb)
np.save("z3,no_parity,eigvectors,"+str(pxp.N),eigvectors_comp_comb)

# z2_overlap = np.log10(np.abs(eigvectors_comp_comb[pxp.keys[z2.ref],:])**2)
# z3_overlap = np.log10(np.abs(eigvectors_comp_comb[pxp.keys[z3.ref],:])**2)

# #top band (scar states)
# from Calculations import get_top_band_indices
# z2_scar_indices = get_top_band_indices(eigvalues_comb,z2_overlap,N,150,300)
# z3_scar_indices = get_top_band_indices(eigvalues_comb,z3_overlap,int(2*N/3),200,300)
# print(np.size(z2_scar_indices))
# print(np.size(z3_scar_indices))
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

# # # #scars in full comp basis
# # # z2_scars = np.zeros(pxp.dim)
# # # for n in range(0,np.size(z2_scar_indices,axis=0)):
    # # # z2_scars = np.vstack((z2_scars,eigvectors_comp_comb[:,z2_scar_indices[n]]))
# # # z2_scars = np.transpose(np.delete(z2_scars,0,axis=0))

# # # z3_scars = np.zeros(pxp.dim)
# # # for n in range(0,np.size(z3_scar_indices,axis=0)):
    # # # z3_scars = np.vstack((z3_scars,eigvectors_comp_comb[:,z3_scar_indices[n]]))
# # # z3_scars = np.transpose(np.delete(z3_scars,0,axis=0))

# # # plt.scatter(eigvalues_comb,z3_overlap)
# # # for n in range(0,np.size(z2_scar_indices,axis=0)):
    # # # plt.scatter(eigvalues_comb[z2_scar_indices[n]],z3_overlap[z2_scar_indices[n]],marker="x",color="red",s=100)
# # # plt.show()
    
# # # plt.scatter(eigvalues_comb,z2_overlap)
# # # for n in range(0,np.size(z3_scar_indices,axis=0)):
    # # # plt.scatter(eigvalues_comb[z3_scar_indices[n]],z2_overlap[z3_scar_indices[n]],marker="x",color="red",s=100)
# # # plt.show()
