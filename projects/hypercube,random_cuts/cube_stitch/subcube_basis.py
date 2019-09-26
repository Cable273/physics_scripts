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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

def krylov(v,H,krylov_size):
    #generate subspace by repeated application of H
    temp = v
    for n in range(0,krylov_size):
        temp = np.dot(H,temp)
        v = np.vstack((v,temp))

    #orthonormalize basis using QR decomposition
    v = np.transpose(v)
    #orthornormalize with qr decomp
    x,r = np.linalg.qr(v)
    return x

def find_hamming_sectors(state_bits,system):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,system.N+1):
        hamming_sectors[n] = []
    for n in range(0,system.dim):
        h = 0
        for m in range(0,system.N,1):
            if system.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],system.basis_refs[n])
    return hamming_sectors

def sublattice_state(v1,v2,system):
    state_bits = np.zeros(system.N)
    for n in range(0,np.size(v1)):
        state_bits[2*v1[n]] = 1
    for n in range(0,np.size(v2)):
        state_bits[2*v2[n]+1] = 1
    return bin_state(state_bits,system)

#pxp cube
N=8
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

#create dictionary + hash table storing refs of all states in perm sector |n,m>
basis_perm_labels = np.zeros((pxp.dim,2))
perm_sector_refs = dict()
def perm_key(v):
    return bin_to_int_base_m(v,int(pxp.N/2)+1)
for n in range(0,int(pxp.N/2)+1):
    for m in range(0,int(pxp.N/2)+1):
        perm_sector_refs[perm_key([n,m])] = []
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    c1=0
    c2=0
    for m in range(0,pxp.N):
        if pxp.basis[n][m] == 1:
            if m % 2 == 0:
                c1 = c1 + 1
            elif m % 2 != 0:
                c2 = c2 + 1
    basis_perm_labels[n] = np.array([c1,c2])
    perm_sector_refs[perm_key(np.array((c1,c2)))] = np.append(perm_sector_refs[perm_key(np.array((c1,c2)))],pxp.basis_refs[n])

#identify all root states
# find perm sectors root states live in
perm_labels = np.zeros(((int(pxp.N/2))**2,2))
temp = np.arange(0,pxp.N/2)
from itertools import *
temp = product(temp,temp)
c=0
for label in temp:
    perm_labels[c] = label
    c=c+1

edge_vertex_sectors = np.zeros(2)
for n in range(0,np.size(perm_labels,axis=0)):
    temp_sum = np.sum(perm_labels[n])
    if temp_sum == int(pxp.N/2-1):
        edge_vertex_sectors = np.vstack((edge_vertex_sectors,perm_labels[n]))
edge_vertex_sectors = np.delete(edge_vertex_sectors,0,axis=0)

root_sectors = np.zeros(2)
for n in range(0,np.size(edge_vertex_sectors,axis=0)-1,2):
    temp = np.zeros(2)
    temp[0] = edge_vertex_sectors[n][0]
    temp[1] = edge_vertex_sectors[n+1][1]
    root_sectors = np.vstack((root_sectors,temp))
root_sectors = np.delete(root_sectors,0,axis=0)

edge_vertex_sectors = np.delete(edge_vertex_sectors,0,axis=0)
edge_vertex_sectors = np.delete(edge_vertex_sectors,np.size(edge_vertex_sectors,axis=0)-1,axis=0)
root_sectors = np.unique(np.vstack((root_sectors,np.flip(root_sectors,axis=1))),axis=0)

#form projectors into edge vertex sector to check root nodes
edge_vertex_projectors = dict()
for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    dim = np.size(perm_sector_refs[perm_key(edge_vertex_sectors[n])])
    edge_vertex_projectors[n] = np.zeros((pxp.dim,pxp.dim))
    ref_list = perm_sector_refs[perm_key(edge_vertex_sectors[n])]
    for m in range(0,np.size(ref_list)):
        index = pxp.keys[ref_list[m]]
        edge_vertex_projectors[n][index,index] = 1

H = spin_Hamiltonian(pxp,"x")
H.gen()
#find root states by those which reside in root sector and are not killed by PH, P a projector into edge_vertex_sectors
# root_refs = []
root_refs = dict()
for n in range(0,np.size(root_sectors,axis=0)):
    root_refs[perm_key(root_sectors[n])] = []
    for m in range(0,np.size(perm_sector_refs[perm_key(root_sectors[n])],axis=0)):
        ref = perm_sector_refs[perm_key(root_sectors[n])][m]
        state = np.dot(H.sector.matrix(),ref_state(ref,pxp).prod_basis())
        for j in range(0,len(edge_vertex_projectors)):
            temp = np.dot(edge_vertex_projectors[j],state)
            if np.abs(np.sum(temp))>1e-5:
                root_refs[perm_key(root_sectors[n])] = np.append(root_refs[perm_key(root_sectors[n])],ref)
                break

for n in range(0,np.size(root_sectors,axis=0)):
    print("\n")
    for m in range(0,np.size(root_refs[perm_key(root_sectors[n])],axis=0)):
        ref = root_refs[perm_key(root_sectors[n])][m]
        print(pxp.basis[pxp.keys[ref]],basis_perm_labels[pxp.keys[ref]])

# smaller hypercubes + hamming, for subcube identification
sub_cube_systems = dict()
sub_cube_hamming = dict()
for n in range(int(pxp.N/2),1,-1):
    sub_cube_systems[n] = unlocking_System([0,1],"open",2,n)
    sub_cube_systems[n].gen_basis()
    z_temp = np.ones(sub_cube_systems[n].N)
    z_temp[np.size(z_temp,axis=0)-1] = 0
    z_temp = bin_state(z_temp,sub_cube_systems[n])
    sub_cube_hamming[n] = find_hamming_sectors(z_temp.bits,sub_cube_systems[n])

#find pos of zeros + ones for subcubes
#loc of poss 1 that is zero
for n in range(0,np.size(root_sectors,axis=0)):
    for m in range(0,np.size(root_refs[perm_key(root_sectors[n])],axis=0)):
        ref = root_refs[perm_key(root_sectors[n])][m]
        state = pxp.basis[pxp.keys[ref]]
        poss_zero_loc = []
        for i in range(0,np.size(state,axis=0)):
            if i % 2 != 0:
                if state[i] == 0:
                    if i == 0:
                        im1 = pxp.N-1
                        ip1 = i + 1
                    elif i == pxp.N-1:
                        im1 = i - 1
                        ip1 = 0
                    else:
                        im1 = i - 1
                        ip1 = i + 1

                    if state[ip1] == 0 and state[im1] == 0:
                        poss_zero_loc = np.append(poss_zero_loc,i)
        print(state,poss_zero_loc)


# #form subcube basis from root states + smaller hypercubes


# #combine subcubes of same size on same side of Hamming graph

# #Orthogonalize
