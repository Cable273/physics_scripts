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
N=10
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
print(edge_vertex_sectors)

root_sectors = np.unique(np.vstack((root_sectors,np.flip(root_sectors,axis=1))),axis=0)
print(root_sectors)
    
H = spin_Hamiltonian(pxp,"x")
H.gen()
#find root states as those connected to edge vertex sectors while also residing in root sector
root_refs = dict()
root_ref_from= dict()
root_ref_from_sector= dict()
for n in range(0,np.size(root_sectors,axis=0)):
    root_refs[perm_key(root_sectors[n])] = []
    root_ref_from[perm_key(root_sectors[n])] = []
    root_ref_from_sector[perm_key(root_sectors[n])] = np.zeros(2)

for n in range(0,np.size(edge_vertex_sectors,axis=0)):
    refs = perm_sector_refs[perm_key(edge_vertex_sectors[n])]
    for m in range(0,np.size(refs,axis=0)):
        state = np.zeros(pxp.dim)
        state[pxp.keys[refs[m]]] = 1

        #find new product states H|psi> maps to
        state = np.dot(H.sector.matrix(),state)
        acted_refs = []
        for i in range(0,np.size(state,axis=0)):
            if state[i] == 1:
                acted_refs = np.append(acted_refs,pxp.basis_refs[i])

        #check those in root sector, these are root states
        in_root_sector = np.zeros(np.size(acted_refs))
        for i in range(0,np.size(acted_refs,axis=0)):
            for j in range(0,np.size(root_sectors,axis=0)):
                if acted_refs[i] in perm_sector_refs[perm_key(root_sectors[j])]:
                    print(edge_vertex_sectors[n],root_sectors[j])
                    root_refs[perm_key(root_sectors[j])] = np.append(root_refs[perm_key(root_sectors[j])],acted_refs[i])
                    root_ref_from[perm_key(root_sectors[j])] = np.append(root_ref_from[perm_key(root_sectors[j])],refs[m])
                    root_ref_from_sector[perm_key(root_sectors[j])] = np.vstack((root_ref_from_sector[perm_key(root_sectors[j])],edge_vertex_sectors[n]))

                    #track which sector maps from, to find 0/1 locations to fill from subcube basis
                    break
                
for n in range(0,np.size(root_sectors,axis=0)):
    refs = root_refs[perm_key(root_sectors[n])]
    print("\n")
    for m in range(0,np.size(refs,axis=0)):
        print(pxp.basis[pxp.keys[refs[m]]],root_sectors[n],root_ref_from[perm_key(root_sectors[n])][m],pxp.basis[pxp.keys[root_ref_from[perm_key(root_sectors[n])][m]]])

#for subcube basis from roots + smaller cube basis
# smaller hypercubes + hamming, for subcube identification

sub_cube_systems = dict()
sub_cube_hamming = dict()
for n in range(int(pxp.N/2),1,-1):
    sub_cube_systems[n] = unlocking_System([0,1],"open",2,n)
    sub_cube_systems[n].gen_basis()
    z_temp = np.ones(sub_cube_systems[n].N)
    if n != int(pxp.N/2):
        z_temp[np.size(z_temp,axis=0)-1] = 0
    z_temp = bin_state(z_temp,sub_cube_systems[n])
    sub_cube_hamming[n] = find_hamming_sectors(z_temp.bits,sub_cube_systems[n])

subcube_basis = dict()
for n in range(0,np.size(root_sectors,axis=0)):
    subcube_basis[perm_key(root_sectors[n])] = dict()
    refs = root_refs[perm_key(root_sectors[n])]
    refs_from = root_ref_from[perm_key(root_sectors[n])]
    refs_from_sector = root_ref_from_sector[perm_key(root_sectors[n])]
    refs_from_sector = np.delete(refs_from_sector,0,axis=0)
    for m in range(0,np.size(refs)):
        subcube_basis[perm_key(root_sectors[n])][m] = np.zeros(pxp.dim)
        cube_dim = int(np.sum(refs_from_sector[m]))
        state_bits = pxp.basis[pxp.keys[refs[m]]]
        state_from_bits = pxp.basis[pxp.keys[refs_from[m]]]
        bit_loc = np.zeros(cube_dim)
        c=0

        #get 1 bit loc from orig state
        for i in range(0,np.size(state_bits,axis=0)):
            if state_bits[i] == 1:
                bit_loc[c] = i
                c = c +1

        # get loc of flippable 0 to 1 from connected sector in hypercube graph
        for i in range(0,np.size(state_from_bits)):
            if state_from_bits[i] == 1:
                if i not in bit_loc:
                    bit_loc[c] = i
                    break

        for i in range(0,len(sub_cube_hamming[cube_dim])):
            sub_refs = sub_cube_hamming[cube_dim][i]
            new_basis_state = np.zeros(pxp.dim)
            for j in range(0,np.size(sub_refs,axis=0)):
                state_bits = sub_cube_systems[cube_dim].basis[sub_cube_systems[cube_dim].keys[sub_refs[j]]]
                temp_bits=np.zeros(pxp.N)
                for k in range(0,np.size(state_bits,axis=0)):
                    temp_bits[int(bit_loc[k])] = state_bits[k]
                temp_ref = bin_to_int_base_m(temp_bits,pxp.base)
                new_basis_state = new_basis_state + ref_state(temp_ref,pxp).prod_basis()
            new_basis_state = new_basis_state / np.power(np.vdot(new_basis_state,new_basis_state),0.5)
            subcube_basis[perm_key(root_sectors[n])][m] = np.vstack((subcube_basis[perm_key(root_sectors[n])][m],new_basis_state))
        subcube_basis[perm_key(root_sectors[n])][m] = np.transpose(np.delete(subcube_basis[perm_key(root_sectors[n])][m],0,axis=0))

#hypercube from Neel/AntiNeel
z0=zm_state(2,1,pxp)
z1=zm_state(2,1,pxp,1)
cube_dim = int(pxp.N/2)
cube_basisL = np.zeros((pxp.dim,len(sub_cube_hamming[cube_dim])))
cube_basisR = np.zeros((pxp.dim,len(sub_cube_hamming[cube_dim])))
for n in range(0,len(sub_cube_hamming[cube_dim])):
    refs = sub_cube_hamming[cube_dim][n]
    temp_stateL = np.zeros(pxp.dim)
    temp_stateR = np.zeros(pxp.dim)
    one_locL = np.arange(0,pxp.N-1,2)
    one_locR = np.arange(1,pxp.N,2)
    for m in range(0,np.size(refs,axis=0)):
        state_bits = sub_cube_systems[cube_dim].basis[sub_cube_systems[cube_dim].keys[refs[m]]]
        temp_bitsL = np.zeros(pxp.N)
        temp_bitsR = np.zeros(pxp.N)
        for i in range(0,np.size(state_bits,axis=0)):
            temp_bitsL[int(one_locL[i])] = state_bits[i]
            temp_bitsR[int(one_locR[i])] = state_bits[i]
        temp_refL = bin_to_int_base_m(temp_bitsL,pxp.base)
        temp_refR = bin_to_int_base_m(temp_bitsR,pxp.base)
        temp_stateL = temp_stateL + ref_state(temp_refL,pxp).prod_basis()
        temp_stateR = temp_stateR + ref_state(temp_refR,pxp).prod_basis()
    temp_stateL = temp_stateL / np.power(np.vdot(temp_stateL,temp_stateL),0.5)
    temp_stateR = temp_stateR / np.power(np.vdot(temp_stateR,temp_stateR),0.5)
    cube_basisL[:,n] = temp_stateL
    cube_basisR[:,n] = temp_stateR

# combine subcube basis (translational symm)
subcube_combined_basis=dict()
for n in range(0,np.size(root_sectors,axis=0)):
    subcube_combined_basis[perm_key(root_sectors[n])] = subcube_basis[perm_key(root_sectors[n])][0]
    for m in range(1,len(subcube_basis[perm_key(root_sectors[n])])):
        subcube_combined_basis[perm_key(root_sectors[n])] = subcube_combined_basis[perm_key(root_sectors[n])] + subcube_basis[perm_key(root_sectors[n])][m]

for n in range(0,np.size(root_sectors,axis=0)):
    for m in range(0,np.size(subcube_combined_basis[perm_key(root_sectors[n])],axis=1)):
        temp = subcube_combined_basis[perm_key(root_sectors[n])][:,m]
        temp = temp / np.power(np.vdot(temp,temp),0.5)
        subcube_combined_basis[perm_key(root_sectors[n])][:,m] = temp


#combine basis
basis = cube_basisL
basis = np.hstack((basis,cube_basisR))
for n in range(0,np.size(root_sectors,axis=0)):
    print(root_sectors[n])
    basis = np.hstack((basis,subcube_combined_basis[perm_key(root_sectors[n])]))
basis = np.hstack((basis,cube_basisR))
basis = np.unique(basis,axis=1)
print(np.size(basis,axis=1))
basis,temp = np.linalg.qr(basis)
from Diagnostics import print_wf
for n in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,n],pxp,1e-2)
    


H.sector.find_eig()

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
e,u = np.linalg.eigh(H_rot)
plt.matshow(np.abs(H_rot))
plt.show()

overlap = np.log10(np.abs(u[0,:])**2)
plt.scatter(e,overlap,marker="s",color="red",s=200,alpha=0.5,label="Subcube basis")

eigenvalues = np.copy(H.sector.eigvalues())
z=zm_state(2,1,pxp)
overlap = eig_overlap(z,H).eval()
# to_del=[]
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-5:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
# plt.scatter(eigenvalues,overlap,label="Exact")
plt.legend()
plt.show()

u_comp_basis = np.dot(basis,u)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(u_comp_basis,axis=1)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.scatter(e,exact_overlap)
plt.show()
