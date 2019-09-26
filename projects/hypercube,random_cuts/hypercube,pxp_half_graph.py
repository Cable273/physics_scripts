#!/usr/bin/env python# -*- coding: utf-8 -*-

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

pxp = unlocking_System([0,1],"periodic",2,12)
pxp.gen_basis()
# pxp_syms = model_sym_data(pxp,[translational(pxp)])

def find_hamming_sectors(state_bits):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,pxp.N+1):
        hamming_sectors[n] = []
    for n in range(0,pxp.dim):
        h = 0
        for m in range(0,pxp.N,1):
            if pxp.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],pxp.basis_refs[n])
    return hamming_sectors

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
to_remove_refs = []
for n in range(int(pxp.N/2)+1,len(hamming_sectors)):
    to_remove_refs = np.append(to_remove_refs,hamming_sectors[n])

for n in range(0,np.size(to_remove_refs,axis=0)):
    if np.abs(to_remove_refs[n] )<1e-10:
        print("ZERO DELETED")

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

H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()
t=np.arange(0,20,0.1)
states=dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    states[n] = ref_state(pxp.basis_refs[n],pxp)
# states[0] = zm_state(2,1,pxp)
# states[1] = ref_state(1,pxp)
# states[1] = zm_state(3,1,pxp)
# states[2] = zm_state(3,1,pxp,1)
# states[3] = zm_state(3,1,pxp,2)
# states[4] = ref_state(0,pxp)
f=dict()
pbar=ProgressBar()
for n in pbar(range(0,len(states))):
    f[n] = fidelity(states[n],H).eval(t,states[n])
for n in range(0,len(f)):
    plt.plot(t,f[n])
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"Half Hypercube Fidelity, N=12")
plt.show()

ent_vals = np.zeros(np.size(pxp.basis_refs))
ent = entropy(pxp)
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    ent_vals[n] = ent.eval(H.sector.eigvectors()[:,n])
plt.scatter(H.sector.eigvalues(),ent_vals)
plt.show()

temp = np.zeros(np.size(t))
print(temp)
for n in range(0,np.size(t,axis=0)):
    temp[n] = f[0][n]+f[4][n]+f[1][n]+f[2][n]+f[3][n]
plt.plot(t,np.log(1-temp))
plt.show()
