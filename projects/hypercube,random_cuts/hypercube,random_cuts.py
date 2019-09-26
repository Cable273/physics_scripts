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
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state


pxp = unlocking_System([0,1],"periodic",2,8)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
# pxp_syms = model_sym_data(pxp,[parity(pxp)])

H=spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H=H.sector.matrix()

p=0.01 #percent of links to remove   
print("Cutting links randomly")
pbar=ProgressBar()
for n in pbar(range(0,pxp.dim)):
    for m in range(0,n):
        if np.abs(H[n,m])>1e-5:
            number = np.random.uniform(0,1)
            if number<=p:
                H[n,m] = 0
                H[m,n] = 0

e,u = np.linalg.eigh(H)

t=np.arange(0,20,0.01)
from Calculations import time_evolve_state

states=dict()
states_energy=dict()
f=dict()
ent=entropy(pxp)
ent_vals = np.zeros(np.size(pxp.basis_refs))
pbar=ProgressBar()
print("Plotting fidelity/entropy")
for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    states[n] = ref_state(pxp.basis_refs[n],pxp)
    states_energy[n] = np.conj(u[pxp.keys[states[n].ref],:])
    f[n] = np.zeros(np.size(t))

    ent_vals[n] = ent.eval(u[:,n])
    for m in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(states_energy[n],e,t[m])
        f[n][m] = np.abs(np.vdot(states_energy[n],evolved_state))**2

    plt.plot(t,f[n],alpha=0.6)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle n(0) \vert n(t) \rangle \vert^2$")
plt.title(r"Hypercube $H=\sum_i X_i$, with $P=$"+str(p)+" chance of dropping $H_{nm}, H_{mn}$. Computational basis fidelities, $N=$"+str(pxp.N))
plt.show()
plt.scatter(e,ent_vals)
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.title(r"Hypercube $H=\sum_i X_i$, with $P=$"+str(p)+" chance of dropping $H_{nm}, H_{mn}$. Eigenstate entropies, $N=$"+str(pxp.N))
plt.show()

#identify reviving states
f_max = np.zeros(len(states))
for n in range(0,len(f)):
    for count in range(0,np.size(f[n],axis=0)):
        if f[n][count]<0.1:
            cut_index = count
            break
    f_max[n] = np.max(f[n][cut_index:])
    # if f_max[n] > 0.3:
        # print(pxp.basis[n])
print(pxp.basis[np.argmax(f_max)])
