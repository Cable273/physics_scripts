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

N = 18
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

e = np.load("./pxp,e,"+str(pxp.N)+".npy")
u = np.load("./pxp,u,"+str(pxp.N)+".npy")
z3_overlap = np.load("./pxp,z3_overlap,"+str(pxp.N)+".npy")
mps_overlap = np.load("./pxp,mps_overlap,"+str(pxp.N)+".npy")

from Calculations import get_top_band_indices
z3_scar_indices = get_top_band_indices(e,z3_overlap,int(2*pxp.N/3),150,300,e_diff = 0.5)
mps_scar_indices = get_top_band_indices(e,mps_overlap,int(2*pxp.N/3),150,300,e_diff = 0.5)

plt.scatter(e,z3_overlap)
for n in range(0,np.size(z3_scar_indices,axis=0)):
    plt.scatter(e[z3_scar_indices[n]],z3_overlap[z3_scar_indices[n]],marker="x",s=100,color="red")
plt.show()
    
plt.scatter(e,mps_overlap)
for n in range(0,np.size(mps_scar_indices,axis=0)):
    plt.scatter(e[mps_scar_indices[n]],mps_overlap[mps_scar_indices[n]],marker="x",s=100,color="red")
plt.show()

M = np.zeros((np.size(z3_scar_indices),np.size(mps_scar_indices)),dtype=complex)
for n in range(0,np.size(M,axis=0)):
    for m in range(0,np.size(M,axis=1)):
        M[n,m] = np.vdot(u[:,z3_scar_indices[n]],u[:,mps_scar_indices[m]])
        print(np.abs(M[n,m]),n,m)
plt.matshow(np.abs(M))
plt.xlabel(r"$m$")
plt.ylabel(r"$n$")
plt.title(r"$M_{nm} = \vert \langle (Z^{3}_{scar})_n \vert (Z^{MPS}_{scar})_m \rangle \vert, PXP, N=$"+str(pxp.N))
plt.colorbar()
plt.show()

#identify mps state scar states which has largest overlap with z3 scar state
mps_scar_highest_indices = np.zeros(np.size(z3_scar_indices))
mps_scar_highest_e = np.zeros(np.size(z3_scar_indices))
mps_scar_highest_overlap = np.zeros(np.size(z3_scar_indices))
mps_scar_max_overlap = np.zeros(np.size(z3_scar_indices))
for n in range(0,np.size(z3_scar_indices,axis=0)):
    max_overlap = 0
    max_index = None
    for m in range(0,np.size(mps_scar_indices,axis=0)):
        temp = np.abs(np.vdot(u[:,z3_scar_indices[n]],u[:,mps_scar_indices[m]]))
        if temp > max_overlap:
            max_overlap = temp
            max_index = mps_scar_indices[m]
    print(max_overlap)
    mps_scar_highest_indices[n] = max_index
    mps_scar_highest_e[n] = e[max_index]
    mps_scar_highest_overlap[n] = mps_overlap[max_index]
    mps_scar_max_overlap[n] = max_overlap

print(mps_scar_highest_e)
to_del=[]
for n in range(0,np.size(mps_overlap,axis=0)):
    if mps_overlap[n] <-10:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    mps_overlap=np.delete(mps_overlap,to_del[n])
    e=np.delete(e,to_del[n])

to_del=[]
for n in range(0,np.size(mps_scar_max_overlap,axis=0)):
    if mps_scar_max_overlap[n] <1e-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    mps_scar_max_overlap=np.delete(mps_scar_max_overlap,to_del[n])
    mps_scar_highest_e=np.delete(mps_scar_highest_e,to_del[n])
    mps_scar_highest_overlap=np.delete(mps_scar_highest_overlap,to_del[n])
    
    
plt.scatter(e,mps_overlap)
plt.scatter(mps_scar_highest_e,mps_scar_highest_overlap,marker="D",color="green",alpha=0.6,s=100,label=r"$K=3$ entangled MPS scar state "+"\n"+"with largest overlap "+"\n"+"with $Z_3$ scar states"+"\n"+"(Missing states zero overlap)")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi_{MPS} \vert E \rangle \vert^2)$")
plt.title(r"$PXP, N=18$")
plt.show()

        
    

