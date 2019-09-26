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
edge_vertex_sectors = np.delete(edge_vertex_sectors,0,axis=0)
edge_vertex_sectors = np.delete(edge_vertex_sectors,np.size(edge_vertex_sectors,axis=0)-1,axis=0)
edge_vertex_sectors = np.vstack((np.array([0,int(pxp.N/2)]),edge_vertex_sectors))
edge_vertex_sectors = np.vstack((edge_vertex_sectors,np.array([int(pxp.N/2),0])))
print(edge_vertex_sectors)

# smaller hypercubes + hamming, for subcube identification
sub_cube_systems = dict()
sub_cube_hamming = dict()
for n in range(int(pxp.N/2),1,-1):
    sub_cube_systems[n] = unlocking_System([0,1],"open",2,n)
    sub_cube_systems[n].gen_basis()
    z_temp = ref_state(np.max(sub_cube_systems[n].basis_refs),sub_cube_systems[n])
    sub_cube_hamming[n] = find_hamming_sectors(z_temp.bits,sub_cube_systems[n])

# form hamming basis for subcubes starting from nodes in edge vertex sectors
root_state_one_loc = dict()
for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    root_refs = perm_sector_refs[perm_key(edge_vertex_sectors[n])]
    root_state_one_loc[perm_key(edge_vertex_sectors[n])] = np.zeros((np.size(root_refs),int(np.sum(edge_vertex_sectors[n]))))
    for m in range(0,np.size(root_refs,axis=0)):
        state = pxp.basis[pxp.keys[root_refs[m]]]
        c=0
        for i in range(0,np.size(state,axis=0)):
            if state[i] == 1:
                root_state_one_loc[perm_key(edge_vertex_sectors[n])][m,c] = i
                c=c+1
            if c >=4:
                break

subcube_basis_refs = dict()
for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    subcube_basis_refs[perm_key(edge_vertex_sectors[n])] = dict()
    root_refs = perm_sector_refs[perm_key(edge_vertex_sectors[n])]
    for m in range(0,np.size(root_refs,axis=0)):
        subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m] = dict()
        cube_dim = int(np.size(root_state_one_loc[perm_key(edge_vertex_sectors[n])],axis=1))
        for i in range(0,len(sub_cube_hamming[cube_dim])):
            subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m][i] = []
            for j in range(0,np.size(sub_cube_hamming[cube_dim][i],axis=0)):
                sub_bit_state = sub_cube_systems[cube_dim].basis[sub_cube_systems[cube_dim].keys[sub_cube_hamming[cube_dim][i][j]]]
                temp_state = np.zeros(pxp.N)
                one_loc = root_state_one_loc[perm_key(edge_vertex_sectors[n])][m]
                for k in range(0,np.size(sub_bit_state,axis=0)):
                    temp_state[int(one_loc[k])] = sub_bit_state[k]
                temp_ref = bin_to_int_base_m(temp_state,pxp.base)
                subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m][i] = np.append(subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m][i],temp_ref)
                # print(subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m][i])

subcube_basis = dict()
for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    subcube_basis[perm_key(edge_vertex_sectors[n])] = dict()
    for m in range(0,np.size(root_refs,axis=0)):
        subcube_basis[perm_key(edge_vertex_sectors[n])][m] = np.zeros((pxp.dim,len(subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m])))
        for k in range(0,len(subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m])):
            refs = subcube_basis_refs[perm_key(edge_vertex_sectors[n])][m][k]
            temp = np.zeros(pxp.dim)
            for i in range(0,np.size(refs,axis=0)):
                temp = temp + ref_state(refs[i],pxp).prod_basis()
            temp = temp / np.power(np.vdot(temp,temp),0.5)
            subcube_basis[perm_key(edge_vertex_sectors[n])][m][:,k] = temp
                
#combine basis states into one array
basis = np.zeros(pxp.dim)
for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    for m in range(0,len(subcube_basis[perm_key(edge_vertex_sectors[n])])):
        basis = np.vstack((basis,np.transpose(subcube_basis[perm_key(edge_vertex_sectors[n])][m])))
basis = np.delete(basis,0,axis=0)
basis = np.transpose(basis)
print(np.size(basis,axis=1))
basis = np.unique(basis,axis=1)
basis, temp = np.linalg.qr(basis)
print(np.size(basis,axis=1))

H = spin_Hamiltonian(pxp,"x")
H.gen()
H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))

plt.matshow(np.abs(H_rot))
plt.show()
e,u = np.linalg.eigh(H_rot)
overlap = np.log10(np.abs(u[0,:])**2)
H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
plt.scatter(e,overlap,marker="s",alpha=0.8,color="red",s=200,label="Subcube Basis")
plt.legend()
plt.show()

u_comp_basis = np.dot(basis,u)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(u_comp_basis,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(pxp.basis_refs,axis=0)):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.scatter(e,exact_overlap)
plt.show()


        
        

    
