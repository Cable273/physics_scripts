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

N =12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H1p = np.zeros((pxp.dim,pxp.dim))
H2p = np.zeros((pxp.dim,pxp.dim))
for n in range(0,np.size(pxp.basis,axis=0)):
    bits = pxp.basis[n]
    for m in range(0,np.size(bits,axis=0)):
        if m == np.size(bits)-1:
            mp1 = 0
        else:
            mp1 = m + 1
        if m == 0:
            mm1 = np.size(bits)-1
            mm1 = m - 1

        #H1p
        if m % 3 == 0 and bits[m] == 1:
            new_bits = np.copy(bits)
            new_bits[m] = 0
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            if new_ref in pxp.basis_refs:
                new_index=pxp.keys[new_ref]
                H1p[new_index,n] = H1p[new_index,n] + 1
        if m % 3 == 1 and bits[m] == 0:
            new_bits = np.copy(bits)
            new_bits[m] = 1
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            if new_ref in pxp.basis_refs:
                new_index=pxp.keys[new_ref]
                H1p[new_index,n] = H1p[new_index,n] + 2

        #H2p
        if m % 3 == 0 and bits[m] == 1:
            new_bits = np.copy(bits)
            new_bits[m] = 0
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            if new_ref in pxp.basis_refs:
                new_index=pxp.keys[new_ref]
                H2p[new_index,n] = H2p[new_index,n] + 1
        if m % 3 == 2 and bits[m] == 0:
            new_bits = np.copy(bits)
            new_bits[m] = 1
            new_ref = bin_to_int_base_m(new_bits,pxp.base)
            if new_ref in pxp.basis_refs:
                new_index=pxp.keys[new_ref]
                H2p[new_index,n] = H2p[new_index,n] + 2
            
H1m = np.conj(np.transpose(H1p))
H2m = np.conj(np.transpose(H2p))
    
H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

z=zm_state(3,1,pxp).prod_basis()
fsa_basis1 = z
fsa_basis2 = z
current_state1 = z
current_state2 = z

dim = int(pxp.N/3)
for n in range(0,dim):
    next_state1 = np.dot(H1p,current_state1)
    next_state2 = np.dot(H2p,current_state2)

    next_state1 = next_state1 / np.power(np.vdot(next_state1,next_state1),0.5)
    next_state2 = next_state2 / np.power(np.vdot(next_state2,next_state2),0.5)

    fsa_basis1 = np.vstack((fsa_basis1,next_state1))
    fsa_basis2 = np.vstack((fsa_basis2,next_state2))

    current_state1 = next_state1
    current_state2 = next_state2
    
fsa_basis = np.transpose(np.vstack((fsa_basis1,fsa_basis2)))
from Calculations import gram_schmidt
gs = gram_schmidt(fsa_basis)
gs.ortho()
fsa_basis = gs.ortho_basis

z=zm_state(3,1,pxp)
eig_overlap(z,H).plot()

H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e,u = np.linalg.eigh(H_fsa)
plt.scatter(e,np.log(np.abs(u[0,:])**2),label="FSA",marker="x")
plt.legend()
plt.show()
