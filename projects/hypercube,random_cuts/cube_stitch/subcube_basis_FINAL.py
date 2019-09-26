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

def perm_key(v):
    return bin_to_int_base_m(v,int(pxp.N/2)+1)

#pxp cube
N=10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp)])

H=spin_Hamiltonian(pxp,"x")
H.gen()

#init
sector_refs = dict()
for n in range(0,int(pxp.N/2)+1):
    for m in range(0,int(pxp.N/2)+1):
        sector_refs[perm_key([n,m])] = []

from_sector = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    c1=0
    c2=0
    for m in range(0,pxp.N):
        if pxp.basis[n,m] !=0:
            if m % 2 == 0:
                c1 = c1+1
            else:
                c2 = c2+1
    from_sector[pxp.basis_refs[n]] = np.array([c1,c2])
    sector_refs[perm_key(from_sector[pxp.basis_refs[n]])] = np.append(sector_refs[perm_key(from_sector[pxp.basis_refs[n]])],pxp.basis_refs[n])

#finding mapping to and from sectors
to_sectors = dict()
for n in range(0,pxp.dim):
    non_zero_mapping = []
    for m in range(0,np.size(H.sector.matrix()[:,n],axis=0)):
        if np.abs(H.sector.matrix()[m,n])>1e-5:
            non_zero_mapping = np.append(non_zero_mapping,pxp.basis_refs[m])
    maps_to = np.zeros((np.size(non_zero_mapping),2))
    for m in range(0,np.size(maps_to,axis=0)):
        maps_to[m] = from_sector[int(non_zero_mapping[m])]
    to_sectors[pxp.basis_refs[n]] = np.unique(maps_to,axis=0)
        
    

subcube_rootsL = dict()
sector_labelsL = np.arange(pxp.N/2-2,1,-1)
root_sectorsL = np.zeros((np.size(sector_labelsL),2))
for n in range(0,np.size(sector_labelsL,axis=0)):
    root_sectorsL[n] = np.array([sector_labelsL[n],n])

subcube_sector_sequenceL = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    subcube_sector_sequenceL[perm_key(root_sectorsL[n])] = dict()
    subcube_sector_sequenceL[perm_key(root_sectorsL[n])][0] = root_sectorsL[n]

    r_cap = root_sectorsL[n,1]+1
    r_min = root_sectorsL[n,1]
    dim = int(root_sectorsL[n,0] + 2)

    #do first step manually
    new_sectors = np.zeros((2,2))
    new_sectors[0] = root_sectorsL[n] + np.array([0,1])
    new_sectors[1] = root_sectorsL[n] - np.array([1,0])
    subcube_sector_sequenceL[perm_key(root_sectorsL[n])][1] = new_sectors
    for m in range(2,dim):
        sectors = subcube_sector_sequenceL[perm_key(root_sectorsL[n])][m-1]
        new_sectors = np.zeros(2)
        for k in range(0,np.size(sectors,axis=0)):
            if sectors[k,1] + 1 <=r_cap:
                temp = sectors[k] + np.array([0,1])
                new_sectors = np.vstack((new_sectors,temp))
            if sectors[k,0] - 1 >= 0:
                temp = sectors[k] - np.array([1,0])
                new_sectors = np.vstack((new_sectors,temp))
        new_sectors = np.unique(np.delete(new_sectors,0,axis=0),axis=0)
        subcube_sector_sequenceL[perm_key(root_sectorsL[n])][m] = new_sectors

subcube_rootsR = dict()
sector_labelsR = np.arange(pxp.N/2-2,1,-1)
root_sectorsR = np.zeros((np.size(sector_labelsR),2))
for n in range(0,np.size(sector_labelsR,axis=0)):
    root_sectorsR[n] = np.array([n,sector_labelsL[n]])
subcube_sector_sequenceR = dict()
for n in range(0,np.size(root_sectorsR,axis=0)):
    subcube_sector_sequenceR[perm_key(root_sectorsR[n])] = dict()
    for m in range(0,len(subcube_sector_sequenceL[perm_key(root_sectorsL[n])])):
        sector_seq = subcube_sector_sequenceL[perm_key(root_sectorsL[n])][m]
        subcube_sector_sequenceR[perm_key(root_sectorsR[n])][m] = np.flip(subcube_sector_sequenceL[perm_key(root_sectorsL[n])][m])

    # print("\n")
    # for m in range(0,len(subcube_sector_sequenceR[perm_key(root_sectorsR[n])])):
        # print(subcube_sector_sequenceR[perm_key(root_sectorsR[n])][m])

    # print("\n")
    # for m in range(0,len(subcube_sector_sequenceL[perm_key(root_sectorsL[n])])):
        # print(subcube_sector_sequenceL[perm_key(root_sectorsL[n])][m])

def find_sector_roots(root_refs_pre_trim,target_sector):
    root_refs = []
    for m in range(0,np.size(root_refs_pre_trim,axis=0)):
        next_sector = to_sectors[root_refs_pre_trim[m]]
        missing_sector = 0
        for k in range(0,np.size(target_sector,axis=0)):
            if (target_sector[k] == next_sector).all(1).any()==False:
            # if target_sector[k] not in next_sector:
                missing_sector = 1
                break
        if missing_sector == 0:
            root_refs = np.append(root_refs,root_refs_pre_trim[m])
    return root_refs

subcube_basisL = dict()
subcube_basisR = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    dim = int(root_sectorsL[n,0]+2)
    subcube_basisL[perm_key(root_sectorsL[n])] = dict()
    subcube_basisR[perm_key(root_sectorsR[n])] = dict()

    root_refs_pre_trimL = sector_refs[perm_key(root_sectorsL[n])]
    root_refs_pre_trimR = sector_refs[perm_key(root_sectorsR[n])]
    target_sectorL = subcube_sector_sequenceL[perm_key(root_sectorsL[n])][1]
    target_sectorR = subcube_sector_sequenceR[perm_key(root_sectorsR[n])][1]
    root_refsL = find_sector_roots(root_refs_pre_trimL,target_sectorL)
    root_refsR = find_sector_roots(root_refs_pre_trimR,target_sectorR)
    print(root_refsL,root_refsR)

    for m in range(0,np.size(root_refsL,axis=0)):
        subcube_basisL[perm_key(root_sectorsL[n])][m] = np.zeros((pxp.dim,dim))
        subcube_basisR[perm_key(root_sectorsR[n])][m] = np.zeros((pxp.dim,dim))

        #first state ,root node
        subcube_basisL[perm_key(root_sectorsL[n])][m][pxp.keys[root_refsL[m]],0] = 1
        subcube_basisR[perm_key(root_sectorsR[n])][m][pxp.keys[root_refsR[m]],0] = 1
        current_layer_refsL = [root_refsL[m]]
        current_layer_refsR = [root_refsR[m]]
        next_layer_refsL = []
        next_layer_refsR = []

        #loop through rest of each hamming layer
        for k in range(1,dim):
            target_sectorL = subcube_sector_sequenceL[perm_key(root_sectorsL[n])][k]
            temp_stateL = np.zeros(pxp.dim)
            for i in range(0,np.size(current_layer_refsL,axis=0)):
                #act on root state with H
                stateL = H.sector.matrix()[:,pxp.keys[current_layer_refsL[i]]]

                #see which product states it maps to
                maps_toL = []
                for u in range(0,np.size(stateL,axis=0)):
                    if np.abs(stateL[u])>1e-5:
                        maps_toL = np.append(maps_toL,pxp.basis_refs[u])
                    
                #see which of these states map to target perm sector
                for u in range(0,np.size(maps_toL,axis=0)):
                    state_from = from_sector[maps_toL[u]]
                    if (state_from == target_sectorL).all(1).any():
                        next_layer_refsL = np.append(next_layer_refsL,maps_toL[u])
                        temp_stateL[pxp.keys[maps_toL[u]]] = 1

            temp_stateL = temp_stateL / np.power(np.vdot(temp_stateL,temp_stateL),0.5)
            subcube_basisL[perm_key(root_sectorsL[n])][m][:,k] = temp_stateL
            next_layer_refsL = np.unique(next_layer_refsL)
            current_layer_refsL = next_layer_refsL
            next_layer_refsL  = []

        for k in range(1,dim):
            target_sectorR = subcube_sector_sequenceR[perm_key(root_sectorsR[n])][k]
            temp_stateR = np.zeros(pxp.dim)
            for i in range(0,np.size(current_layer_refsR,axis=0)):
                #act on root state with H
                stateR = H.sector.matrix()[:,pxp.keys[current_layer_refsR[i]]]

                #see which product states it maps to
                maps_toR = []
                for u in range(0,np.size(stateR,axis=0)):
                    if np.abs(stateR[u])>1e-5:
                        maps_toR = np.append(maps_toR,pxp.basis_refs[u])

                for u in range(0,np.size(maps_toR,axis=0)):
                    state_from = from_sector[maps_toR[u]]
                    # if state_from in target_sectorR:
                    if (state_from == target_sectorR).all(1).any():
                        next_layer_refsR = np.append(next_layer_refsR,maps_toR[u])
                        temp_stateR[pxp.keys[maps_toR[u]]] = 1
            
            temp_stateR = temp_stateR / np.power(np.vdot(temp_stateR,temp_stateR),0.5)
            subcube_basisR[perm_key(root_sectorsR[n])][m][:,k] = temp_stateR
            next_layer_refsR = np.unique(next_layer_refsR)
            current_layer_refsR = next_layer_refsR
            next_layer_refsR  = []

#combine subcube basis
subcube_combined_basisL = dict()
subcube_combined_basisR = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    subcube_combined_basisL[perm_key(root_sectorsL[n])] = subcube_basisL[perm_key(root_sectorsL[n])][0]
    subcube_combined_basisR[perm_key(root_sectorsR[n])] = subcube_basisR[perm_key(root_sectorsR[n])][0]
    for m in range(1,len(subcube_basisL[perm_key(root_sectorsL[n])])):
        subcube_combined_basisL[perm_key(root_sectorsL[n])] = subcube_combined_basisL[perm_key(root_sectorsL[n])] + subcube_basisL[perm_key(root_sectorsL[n])][m]
        subcube_combined_basisR[perm_key(root_sectorsR[n])] = subcube_combined_basisR[perm_key(root_sectorsR[n])] + subcube_basisR[perm_key(root_sectorsR[n])][m]

# #renormalize
for n in range(0,np.size(root_sectorsL,axis=0)):
    for m in range(0,np.size(subcube_combined_basisL[perm_key(root_sectorsL[n])],axis=1)):
        tempL = subcube_combined_basisL[perm_key(root_sectorsL[n])][:,m]
        tempR = subcube_combined_basisR[perm_key(root_sectorsR[n])][:,m]
        tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
        tempR = tempR / np.power(np.vdot(tempR,tempR),0.5)
        subcube_combined_basisL[perm_key(root_sectorsL[n])][:,m] = tempL
        subcube_combined_basisR[perm_key(root_sectorsR[n])][:,m] = tempR

neel_subcube_system = unlocking_System([0,1],"open",2,int(pxp.N/2))
neel_subcube_system.gen_basis()
z=ref_state(np.max(neel_subcube_system.basis_refs),neel_subcube_system)
neel_hamming = find_hamming_sectors(z.bits,neel_subcube_system)

#hypercube from Neel/AntiNeel
z0=zm_state(2,1,pxp)
z1=zm_state(2,1,pxp,1)
cube_dim = int(pxp.N/2)
cube_basisL = np.zeros(pxp.dim)
cube_basisR = np.zeros(pxp.dim)
for n in range(0,len(neel_hamming)):
# for n in range(0,2):
    refs = neel_hamming[n]
    temp_stateL = np.zeros(pxp.dim)
    temp_stateR = np.zeros(pxp.dim)
    one_locL = np.arange(0,pxp.N-1,2)
    one_locR = np.arange(1,pxp.N,2)
    for m in range(0,np.size(refs,axis=0)):
        state_bits = neel_subcube_system.basis[neel_subcube_system.keys[refs[m]]]
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
    # cube_basisL[:,n] = temp_stateL
    # cube_basisR[:,n] = temp_stateR
    cube_basisL = np.vstack((cube_basisL,temp_stateL))
    cube_basisR = np.vstack((cube_basisR,temp_stateR))
cube_basisL = np.transpose(np.delete(cube_basisL,0,axis=0))
cube_basisR = np.transpose(np.delete(cube_basisR,0,axis=0))

#combine total subsector basis
basis = cube_basisL
# print(root_sectorsL)
# print(root_sectorsR)
for n in range(0,np.size(root_sectorsL,axis=0)):
    basis = np.hstack((basis,subcube_combined_basisL[perm_key(root_sectorsL[n])]))
for n in range(0,np.size(root_sectorsR,axis=0)):
    basis = np.hstack((basis,subcube_combined_basisR[perm_key(root_sectorsR[n])]))
basis = np.hstack((basis,cube_basisR))

from Diagnostics import print_wf
for n in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,n],pxp,1e-2)
    
# project Hamiltonian + dynamics
print(np.shape(basis))
basis = np.unique(basis,axis=1)
basis,temp = np.linalg.qr(basis)
print(np.shape(basis))
H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
plt.matshow(np.abs(H_rot))
plt.show()
e,u = np.linalg.eigh(H_rot)
overlap = np.log10(np.abs(u[0,:])**2)

H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
plt.scatter(e,overlap,marker="s",s=200,alpha=0.6,color="red")
plt.show()

u_comp_basis = np.dot(basis,u)
exact_overlap = np.zeros(np.size(u_comp_basis,axis=1))
for n in range(0,np.size(u_comp_basis,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
print(exact_overlap)
plt.scatter(e,exact_overlap)
plt.show()
