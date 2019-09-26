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


pxp = unlocking_System([0,1],"periodic",2,10)
pxp.gen_basis()

#organize states via hamming distance from Neel
hamming_sectors = dict()
for n in range(0,pxp.N+1):
    hamming_sectors[n] = []

for n in range(0,pxp.dim):
    h = 0
    for m in range(0,pxp.N-1,2):
        if pxp.basis[n][m] != 1:
            h = h+1
        if pxp.basis[n][m+1] != 0:
            h = h+1
    hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],pxp.basis_refs[n])

#remove uniformly
# p=0.1 #percent of links to remove   
# to_remove = []
# for n in range(0,pxp.dim):
    # number = np.random.uniform(0,1)
    # if number<=p:
        # to_remove = np.append(to_remove,n)

#remove from sites, ham=L/2 away from Neel
L = int(pxp.N/2)
p = 0.15
to_remove_refs = []
middle_sectors = np.array([L-1,L,L+1])
for sector in middle_sectors:
    for n in range(0,np.size(hamming_sectors[sector])):
        number = np.random.uniform(0,1)
        if number <= p:
            to_remove_refs = np.append(to_remove_refs,hamming_sectors[sector][n])

for n in range(0,np.size(to_remove_refs,axis=0)):
    print(pxp.basis[pxp.keys[to_remove_refs[n]]])

#redo basis
pxp.basis_refs_new = np.zeros(np.size(pxp.basis_refs)-np.size(to_remove_refs))
c=0
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    if pxp.basis_refs[n] not in to_remove_refs:
        pxp.basis_refs_new[c] = pxp.basis_refs[n]
        c = c+1
pxp.basis_refs = pxp.basis_refs_new

pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
for n in range(0,np.size(pxp.basis_refs)):
    pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
pxp.keys = dict()
for n in range(0,np.size(pxp.basis_refs)):
    pxp.keys[int(pxp.basis_refs[n])] = n
    
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
# # pxp_syms = model_sym_data(pxp,[parity(pxp)])

H=spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()
e,u = H.sector.eigvalues(),H.sector.eigvectors()
# H=H.sector.matrix()

# e,u = np.linalg.eigh(H)
t=np.arange(0,10,0.1)
from Calculations import time_evolve_state
z=zm_state(2,1,pxp)
# k = pxp_syms.find_k_ref(z.ref)
# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])

# fidelity(z,H,"use sym").plot(np.arange(0,20,0.01),z)
fidelity(z,H).plot(np.arange(0,20,0.01),z)
plt.show()

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
plt.title(r"Hypercube $H=\sum_i X_i$, with $P=$"+str(p)+" chance of removing state from $L/2-1, L/2, L/2+1$ hamming sector. Computational basis fidelities, $N=$"+str(pxp.N))
plt.show()
plt.scatter(e,ent_vals)
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.title(r"Hypercube $H=\sum_i X_i$, with $P=$"+str(p)+" chance of removing state from $L/2-1, L/2, L/2+1$ hamming sector. Eigenstate entropies, $N=$"+str(pxp.N))
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
