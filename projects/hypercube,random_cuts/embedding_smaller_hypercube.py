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

pxp = unlocking_System([0],"periodic",2,6)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp)])

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

def get_state_connections(state_ref,h_sector,H_matrix):
    pk = 0
    qk = 0
    matrix_row = H_matrix[pxp.keys[state_ref],:]
    for n in range(0,np.size(matrix_row,axis=0)):
        if np.abs(matrix_row[n])>1e-5:
            ref = pxp.basis_refs[n]
            if ref in hamming_sectors[h_sector-1]:
                qk = qk+1
            elif ref in hamming_sectors[h_sector+1]:
                pk = pk+1
    return qk,pk

def find_forward_refs(state_ref,h_sector,H_matrix):
    matrix_row = H_matrix[pxp.keys[state_ref],:]
    forward_refs = []
    for n in range(0,np.size(matrix_row,axis=0)):
        if np.abs(matrix_row[n])>1e-5:
            ref = pxp.basis_refs[n]
            if ref in hamming_sectors[h_sector+1]:
                forward_refs = np.append(forward_refs,ref)
    return forward_refs

# z=ref_state(0,pxp)
# z=zm_state(2,1,pxp)
# hamming_sectors = find_hamming_sectors(z.bits)
# to_remove_refs = []
# for n in range(int(pxp.N/2)+1,len(hamming_sectors)):
    # to_remove_refs = np.append(to_remove_refs,hamming_sectors[n])
# #redo basis
# pxp.basis_refs_new = np.zeros(np.size(pxp.basis_refs)-np.size(to_remove_refs))
# c=0
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # if pxp.basis_refs[n] not in to_remove_refs:
        # pxp.basis_refs_new[c] = pxp.basis_refs[n]
        # c = c+1
# pxp.basis_refs = pxp.basis_refs_new
# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
# pxp.keys = dict()
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.keys[int(pxp.basis_refs[n])] = n
# pxp.dim=np.size(pxp.basis_refs)

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
# z=ref_state(0,pxp)
    
#count non empty sectors
no_sectors=0
for n in range(0,len(hamming_sectors)):
    if np.size(hamming_sectors[n])!=0:
        no_sectors = no_sectors+1

for n in range(0,len(hamming_sectors)-1):
    print("\n")
    for m in range(0,np.size(hamming_sectors[n],axis=0)):
        print(pxp.basis[pxp.keys[hamming_sectors[n][m]]])
    
print(pxp.N,no_sectors)
H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()

qk = dict()
pk = dict()
nk = np.zeros(len(hamming_sectors))
nk[0] = 1
nk[np.size(nk)-1] = 1
for n in range(1,no_sectors):
    qk[n] = np.zeros(np.size(hamming_sectors[n]))
    pk[n] = np.zeros(np.size(hamming_sectors[n]))
    nk[n] = np.size(hamming_sectors[n])
    for m in range(0,np.size(hamming_sectors[n])):
        qk[n][m], pk[n][m] = get_state_connections(hamming_sectors[n][m],n,H.sector.matrix())

pk_node = dict()
qk_node = dict()
for n in range(1,no_sectors):
    for m in range(0,np.size(hamming_sectors[n])):
        ref = hamming_sectors[n][m]
        pk_node[ref] = pk[n][m]
        qk_node[ref] = qk[n][m]

# z = zm_state(2,1,pxp)
pk_node[z.ref] = pxp.N/2    
qk_node[z.ref] = 0
# z1 = zm_state(2,1,pxp,1)
# pk_node[z1.ref] = 0
# qk_node[z1.ref] = int(pxp.N/2)

forward_maps = dict()
for n in range(0,no_sectors-1):
    for m in range(0,np.size(hamming_sectors[n])):
        forward_maps[hamming_sectors[n][m]] = find_forward_refs(hamming_sectors[n][m],n,H.sector.matrix())

#recursively generate all paths and pk sequences. Find those matching smaller hypercube
paths = [[hamming_sectors[0][0]]]
        
print("Generating all paths")
pbar=ProgressBar()
for n in pbar(range(0,no_sectors-1)):
    new_paths = np.zeros((1,np.size(paths,axis=1)+1))
    for m in range(0,np.size(paths,axis=0)):
        current_length = int(np.size(paths,axis=1))
        if current_length == 1:
            forward_states = forward_maps[paths[m][0]]
        else:
            forward_states = forward_maps[paths[m,current_length-1]]
        for k in range(0,np.size(forward_states,axis=0)):
            new_paths = np.vstack((new_paths,np.append(paths[m],[forward_states[k]])))
    new_paths = np.delete(new_paths,0,axis=0)
    paths = new_paths

path_pk_seq = np.zeros((np.size(paths,axis=0),np.size(paths,axis=1)))
path_qk_seq = np.zeros((np.size(paths,axis=0),np.size(paths,axis=1)))
for path_index in range(0,np.size(paths,axis=0)):
    for n in range(0,np.size(paths[path_index])):
        path_pk_seq[path_index,n] = pk_node[int(paths[path_index,n])]
        path_qk_seq[path_index,n] = qk_node[int(paths[path_index,n])]
unique_pk_seq,pk_freq = np.unique(path_pk_seq,axis=0,return_counts=True)
unique_qk_seq,qk_freq = np.unique(path_qk_seq,axis=0,return_counts=True)
print("pk seq")
for n in range(0,np.size(unique_pk_seq,axis=0)):
    print(unique_pk_seq[n],pk_freq[n])
print("qk seq")
for n in range(0,np.size(unique_qk_seq,axis=0)):
    print(unique_qk_seq[n],qk_freq[n])

# opt_pk = np.zeros(len(hamming_sectors)-2)
# c=0
# for n in range(1,len(hamming_sectors)-1):
    # opt_pk[c] = np.power(nk[n+1]/nk[n] * (n+1)*(pxp.N-n),0.5)
    # c=c+1
 
# print("nk's")
# print(nk)
# print("Optimal pk")
# print(opt_pk)
# print("Constrained Hypercube pk")
# print("Pk")
# for n in range(1,len(hamming_sectors)-1):
    # print(pk[n])
# # print("Qk")
# # for n in range(1,len(hamming_sectors)-1):
    # # print(qk[n])
# betas = []
# for n in range(1,len(hamming_sectors)-1):
    # betas = np.append(betas,np.power(1/nk[n],0.5))
# print("Betas")
# print(betas)
# # plt.plot(betas)
# # plt.show()
