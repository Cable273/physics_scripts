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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,inversion
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state
import itertools

def is_zero(a,tol):
    return (np.abs(a)<tol).all()

def print_wf(state,system):
    for n in range(0,np.size(state,axis=0)):
        if np.abs(state[n])>1e-5:
            print(state[n],system.basis[n])

#Form a state made from the polarized state |000...> with n singlets added.
#Form equal superposition over all possible ways of adding n singlets.
#Final state will be a |J,-J> state, with J=L/2 - n
system = unlocking_System([0,1],"periodic",2,4)

n=2 #no singlets to add

sites=np.arange(0,system.N)
#combinatrocs, all possible site indexes to modify 2n spins, to insert n singlets
singlet_sites = np.array((list(itertools.combinations(sites,2*n))))

#of these sites, form all possible pairs (all possible singlet couplings)
no_pairs = int(np.size(singlet_sites,axis=1)/2)

#For each pair, two ways to insert a singlet. either the first index has spin
#up, second down or vice versa. Label as 0,1; find all poss ways unique up to inversion
pair_basis_system = unlocking_System([0,1],"periodic",2,no_pairs)
if no_pairs > 1:
    pair_sym = model_sym_data(pair_basis_system,[inversion(pair_basis_system)])
    unique_pair_refs = pair_sym.find_block_refs([0])
    unique_pair_bits = np.zeros((np.size(unique_pair_refs),no_pairs),dtype=int)
    for n in range(0,np.size(unique_pair_refs,axis=0)):
        unique_pair_bits[n] = pair_basis_system.basis[pair_basis_system.keys[unique_pair_refs[n]]]
else:
    unique_pair_bits = np.array([[0]])
# print(unique_pair_bits)
    
ref_pairs = np.zeros(2)
from combinatorics import all_pairs
for i in range(0,np.size(singlet_sites,axis=0)):
    # print(" ")
    for pair in all_pairs(list(singlet_sites[i])):
        print(pair)
        set_of_pairs = np.array((pair))
        first_spin_sites = set_of_pairs[:,0]
        second_spin_sites = set_of_pairs[:,1]
        for j in range(0,np.size(unique_pair_bits,axis=0)):
            state1 = np.zeros(system.N)
            state2 = np.zeros(system.N)
            # print(unique_pair_bits[j])
            for k in range(0,np.size(first_spin_sites,axis=0)):
                # state1[first_spin_sites[k]] = unique_pair_bits[j,k]
                # state1[second_spin_sites[k]] = 1-unique_pair_bits[j,k]

                # state2[second_spin_sites[k]] = unique_pair_bits[j,k]
                # state2[first_spin_sites[k]] = 1-unique_pair_bits[j,k]

                state1[first_spin_sites[k]] = 1
                state1[second_spin_sites[k]] = 0

                state2[second_spin_sites[k]] = 1
                state2[first_spin_sites[k]] = 0
            print(state1)
            print(state2)
            ref1 = bin_to_int_base_m(state1,system.base)
            ref2 = bin_to_int_base_m(state2,system.base)
            # print(ref1,ref2)
            ref_pairs = np.vstack((ref_pairs,np.array((ref1,ref2))))

ref_pairs = np.delete(ref_pairs,0,axis=0)
print(ref_pairs)
ref_pairs_uniq = np.unique(ref_pairs,axis=0)
ref_pairs_uniq = ref_pairs

print(np.size(ref_pairs,axis=0))
print(np.size(ref_pairs_uniq,axis=0))

psi = np.zeros(np.size(system.basis_refs))
for n in range(0,np.size(ref_pairs_uniq,axis=0)):
    psi[system.keys[ref_pairs_uniq[n,0]]] = psi[system.keys[ref_pairs_uniq[n,0]]] + 1
    psi[system.keys[ref_pairs_uniq[n,1]]] = psi[system.keys[ref_pairs_uniq[n,1]]] - 1
print(psi)
psi = psi / np.power(np.vdot(psi,psi),0.5)
# print("HI")
print_wf(psi,system)

# psi = np.zeros(np.size(system.basis_refs))
# psi[5] = 3
# psi[3] = 2
# psi[10] = -3
# psi[6] = 1
# psi[9] = -1
# psi[12] = -2
# psi = psi/np.power(np.vdot(psi,psi),0.5)

X=spin_Hamiltonian(system,"x")
Y=spin_Hamiltonian(system,"y")
Z=spin_Hamiltonian(system,"z")
X.gen()
Y.gen()
Z.gen()
S2 = np.dot(X.sector.matrix(),X.sector.matrix())+np.dot(Y.sector.matrix(),Y.sector.matrix())+np.dot(Z.sector.matrix(),Z.sector.matrix())

S2_exp = np.real(np.vdot(psi,np.dot(S2,psi)))
s_exp = 0.5*(-1+np.power(1+4*S2_exp,0.5))
z_exp = np.real(np.vdot(psi,np.dot(Z.sector.matrix(),psi)))
print("J max=",0.5*system.N)
print("S^2 exp=",S2_exp)
print("J=",s_exp)
print("Jz=",z_exp)

print("S^2 eig? ",is_zero(np.dot(S2,psi)-S2_exp*psi,1e-5))
print("Z eig? ",is_zero(np.dot(Z.sector.matrix(),psi)-z_exp*psi,1e-5))

# e,u = np.linalg.eigh(S2)
# print(0.5*(-1+np.power(1+4*e,0.5)))
# # print(u)

        
# print("state 0")
# print_wf(u[:,0],system)
# print(np.real(np.vdot(u[:,0],np.dot(Z.sector.matrix(),u[:,0]))))
# print("state 1")
# print_wf(u[:,1],system)
# print(np.real(np.vdot(u[:,1],np.dot(Z.sector.matrix(),u[:,1]))))
# print("Equal sup")
# psi2 = np.power(2,-0.5)*(u[:,0] + u[:,1])
# print_wf(psi2,system)
