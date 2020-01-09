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

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

H0 = spin_Hamiltonian(pxp,"x",pxp_syms)

V1 = Hamiltonian(pxp,pxp_syms)
V1.site_ops[1] = np.array([[0,1],[1,0]])
V1.model=np.array([[0,1,0,0],[0,0,1,0]])
V1.model_coef = np.array([1,1])

V2 = Hamiltonian(pxp,pxp_syms)
V2.site_ops[1] = np.array([[0,1],[1,0]])
V2.model=np.array([[0,1,1,1,0]])
V2.model_coef = np.array([1])

H0.gen()
V1.gen()
V2.gen()


H1 = H_operations.add(H0,V1,np.array([1,0.108]))
H2 = H_operations.add(H0,V2,np.array([1,0.122]))
H1.sector.find_eig()
H2.sector.find_eig()

z=zm_state(2,1,pxp)
overlap1 = eig_overlap(z,H1).eval()
overlap2 = eig_overlap(z,H1).eval()

from Calculations import get_top_band_indices
scar_indices_ppxp = get_top_band_indices(H1.sector.eigvalues(),overlap1,pxp.N,150,300,e_diff = 0.5)
scar_indices_pxxxp = get_top_band_indices(H2.sector.eigvalues(),overlap2,pxp.N,150,300,e_diff = 0.5)

plt.scatter(H1.sector.eigvalues(),overlap1)
for n in range(0,np.size(scar_indices_ppxp,axis=0)):
    plt.scatter(H1.sector.eigvalues()[scar_indices_ppxp[n]],overlap1[scar_indices_ppxp[n]],marker="x",color="red")
plt.show()

plt.scatter(H2.sector.eigvalues(),overlap2)
for n in range(0,np.size(scar_indices_ppxp,axis=0)):
    plt.scatter(H2.sector.eigvalues()[scar_indices_ppxp[n]],overlap2[scar_indices_ppxp[n]],marker="x",color="red")
plt.show()

M = np.zeros((np.size(scar_indices_ppxp),np.size(scar_indices_pxxxp)),dtype=complex)
for n in range(0,np.size(M,axis=0)):
    for m in range(0,np.size(M,axis=0)):
        M[n,m] = np.vdot(H1.sector.eigvectors()[:,scar_indices_ppxp[n]],H2.sector.eigvectors()[:,scar_indices_pxxxp[m]])
print(np.diag(M))
plt.matshow(np.abs(M))
plt.colorbar()
plt.show()
u,s,vh = np.linalg.svd(M)
print(np.sum(s))

ppxp_eig_largest_pxxxp_scar_overlap = np.zeros(np.size(scar_indices_pxxxp))
ppxp_eig_largest_pxxxp_scar_indices = np.zeros(np.size(scar_indices_pxxxp))
scar_e = np.zeros(np.size(scar_indices_pxxxp))
scar_o = np.zeros(np.size(scar_indices_pxxxp))
max_overlaps = np.zeros(np.size(scar_indices_pxxxp))
for n in np.arange(0,np.size(scar_indices_pxxxp,axis=0)):
    max_overlap = 0
    max_index = None
    for m in np.arange(0,np.size(H1.sector.eigvectors(),axis=1)):
        temp = np.abs(np.vdot(H1.sector.eigvectors()[:,m],H2.sector.eigvectors()[:,scar_indices_pxxxp[n]]))**2
        if temp > max_overlap:
            max_overlap = temp
            max_index = m
    ppxp_eig_largest_pxxxp_scar_overlap[n] = max_overlap
    ppxp_eig_largest_pxxxp_scar_indices[n] = max_index
    scar_e[n] = H1.sector.eigvalues()[max_index]
    scar_o[n] = overlap1[max_index]
plt.scatter(H1.sector.eigvalues(),overlap1)
plt.scatter(scar_e,scar_o,marker="D",color="red",s=100,alpha=0.6,label="PPXP eigenstates with largest overlap with PXXXP scars")
print(scar_e)
print(scar_o)
print(ppxp_eig_largest_pxxxp_scar_overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle Z_2 \vert E \rangle \vert^2$")
plt.legend()
plt.show()
    

        
    
    
    

