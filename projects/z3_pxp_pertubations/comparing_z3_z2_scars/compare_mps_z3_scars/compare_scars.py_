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
e_nopert = np.load("./pxp,no_pert,e,18.npy")
u_nopert = np.load("./pxp,no_pert,u,18.npy")

e_pert = np.load("./pxp,pert,e,18.npy")
u_pert = np.load("./pxp,pert,u,18.npy")

z=zm_state(3,1,pxp)
overlap_pert = np.log10(np.abs(u_pert[pxp.keys[z.ref],:])**2)
overlap_nopert = np.log10(np.abs(u_nopert[pxp.keys[z.ref],:])**2)

from Calculations import get_top_band_indices
scar_indices_pert = get_top_band_indices(e_pert,overlap_pert,int(2*N/3),150,400,e_diff = 0.5)
scar_indices_nopert = get_top_band_indices(e_nopert,overlap_nopert,int(2*N/3),150,400,e_diff = 0.5)

plt.scatter(e_pert,overlap_pert)
for n in range(0,np.size(scar_indices_pert,axis=0)):
    plt.scatter(e_pert[scar_indices_pert[n]],overlap_pert[scar_indices_pert[n]],marker="x",s=100,color="red")
plt.show()

plt.scatter(e_nopert,overlap_nopert)
for n in range(0,np.size(scar_indices_nopert,axis=0)):
    plt.scatter(e_nopert[scar_indices_nopert[n]],overlap_nopert[scar_indices_nopert[n]],marker="x",s=100,color="red")
plt.show()

M = np.zeros((np.size(scar_indices_nopert,axis=0),np.size(scar_indices_pert,axis=0)),dtype=complex)
for n in range(0,np.size(scar_indices_pert,axis=0)):
    for m in range(0,np.size(scar_indices_nopert,axis=0)):
        M[n,m] = np.vdot(u_pert[:,scar_indices_pert[n]],u_nopert[:,scar_indices_nopert[m]])
plt.matshow(np.abs(M))
plt.xlabel(r"$m$")
plt.ylabel(r"$n$")
plt.title(r"$M_{nm} = \vert \langle (Z^{pert}_{3,scar})_n \vert (Z^{no \, pert}_{3,scar})_m \rangle \vert, PXP, N=$"+str(pxp.N))
plt.colorbar()
plt.show()

#identify perm state scar states which has largest overlap with non perm scar state
no_perm_largest_overlap_indices = np.zeros(np.size(scar_indices_nopert))
no_perm_largest_e = np.zeros(np.size(no_perm_largest_overlap_indices))
no_perm_largest_overlap = np.zeros(np.size(no_perm_largest_overlap_indices))
for n in range(0,np.size(no_perm_largest_overlap_indices,axis=0)):
    max_overlap = 0
    max_index = None
    for m in range(0,np.size(scar_indices_pert,axis=0)):
        temp = np.abs(np.vdot(u_nopert[:,scar_indices_nopert[n]],u_pert[:,scar_indices_pert[m]]))
        if temp > max_overlap:
            max_overlap = temp
            max_index = scar_indices_pert[m]
    no_perm_largest_overlap_indices[n] = max_index
    no_perm_largest_e[n] = e_pert[max_index]
    no_perm_largest_overlap[n] = overlap_pert[max_index]

to_del=[]
for n in range(0,np.size(overlap_pert,axis=0)):
    if overlap_pert[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap_pert=np.delete(overlap_pert,to_del[n])
    e_pert=np.delete(e_pert,to_del[n])
    
plt.scatter(e_pert,overlap_pert)
plt.scatter(no_perm_largest_e,no_perm_largest_overlap,marker="D",color="green",alpha=0.6,s=100,label="Perturbed eigenstates \n with largest overlap \n with unperturbed scar states")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle Z_3 \vert E \rangle \vert^2)$")
plt.title(r"$PXP+\lambda_i V_i, N=$"+str(pxp.N))
plt.show()


        
    
    

