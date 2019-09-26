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
N=8
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
subcube_rootsR = dict()

sector_labels = np.arange(pxp.N/2-2,1,-1)
root_sectorsL = np.zeros((np.size(sector_labels),2))
root_sectorsR = np.zeros((np.size(sector_labels),2))
for n in range(0,np.size(sector_labels,axis=0)):
    root_sectorsL[n] = np.array([sector_labels[n],0])
    root_sectorsR[n] = np.array([0,sector_labels[n]])

#root_refs
root_refsL = dict()
root_refsR = dict()
non_root_refsL = dict()
non_root_refsR = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    root_refsL[perm_key(root_sectorsL[n])] = []
    root_refsR[perm_key(root_sectorsR[n])] = []
    non_root_refsL[perm_key(root_sectorsL[n])] = []
    non_root_refsR[perm_key(root_sectorsR[n])] = []

    refsL = sector_refs[perm_key(root_sectorsL[n])]
    refsR = sector_refs[perm_key(root_sectorsR[n])]

    target_sectorL = root_sectorsL[n] + np.array([0,1])
    target_sectorR = root_sectorsR[n] + np.array([1,0])

    for m in range(0,np.size(refsL,axis=0)):
        #check which sectors it maps to
        mapped_sectorsL = to_sectors[refsL[m]]
        mapped_sectorsR = to_sectors[refsR[m]]
        if target_sectorL in mapped_sectorsL:
            root_refsL[perm_key(root_sectorsL[n])] = np.append(root_refsL[perm_key(root_sectorsL[n])],refsL[m])
        else:
            non_root_refsL[perm_key(root_sectorsL[n])] = np.append(non_root_refsL[perm_key(root_sectorsL[n])],refsL[m])

        if target_sectorR in mapped_sectorsR:
            root_refsR[perm_key(root_sectorsR[n])] = np.append(root_refsR[perm_key(root_sectorsR[n])],refsR[m])
        else:
            non_root_refsR[perm_key(root_sectorsR[n])] = np.append(non_root_refsR[perm_key(root_sectorsR[n])],refsR[m])

non_root_basisL = np.zeros(pxp.dim)
non_root_basisR = np.zeros(pxp.dim)
for n in range(0,np.size(root_sectorsL,axis=0)):
    refsL = non_root_refsL[perm_key(root_sectorsL[n])]
    refsR = non_root_refsR[perm_key(root_sectorsR[n])]
    tempL = np.zeros(pxp.dim)
    tempR = np.zeros(pxp.dim)
    for m in range(0,np.size(refsL,axis=0)):
        tempL = tempL + ref_state(refsL[m],pxp).prod_basis()
        tempR = tempR + ref_state(refsR[m],pxp).prod_basis()
    if np.sum(np.abs(tempL))>1e-5:
        tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
        non_root_basisL = np.vstack((non_root_basisL,tempL))
    if np.sum(np.abs(tempR))>1e-5:
        tempR = tempR / np.power(np.vdot(tempR,tempL),0.5)
        non_root_basisR = np.vstack((non_root_basisR,tempL))
non_root_basisL = np.transpose(np.delete(non_root_basisL,0,axis=0))
non_root_basisR = np.transpose(np.delete(non_root_basisR,0,axis=0))
    
# form subcube basis stemming from root nodes
hamming_layer_refsL = dict()
hamming_layer_refsR = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    print(root_sectorsR[n])
    hamming_layer_refsL[perm_key(root_sectorsL[n])] = dict()
    hamming_layer_refsR[perm_key(root_sectorsR[n])] = dict()

    hamming_layer_refsL[perm_key(root_sectorsL[n])][0] = root_refsL[perm_key(root_sectorsL[n])]
    hamming_layer_refsR[perm_key(root_sectorsR[n])][0] = root_refsR[perm_key(root_sectorsR[n])]

    #first target sector flip 1 bit up
    target_sectorL = root_sectorsL[n] + np.array([0,1])
    target_sectorR = root_sectorsR[n] + np.array([1,0])
    print(target_sectorR)

    target_refsL = sector_refs[perm_key(target_sectorL)]
    target_refsR = sector_refs[perm_key(target_sectorR)]
    kept_refsL = []
    kept_refsR = []
    for m in range(0,np.size(target_refsL,axis=0)):
        if root_sectorsL[n] in to_sectors[target_refsL[m]]:
            kept_refsL = np.append(kept_refsL,target_refsL[m])
        if root_sectorsR[n] in to_sectors[target_refsR[m]]:
            kept_refsR = np.append(kept_refsR,target_refsR[m])

    hamming_layer_refsL[perm_key(root_sectorsL[n])][1] = kept_refsL
    hamming_layer_refsR[perm_key(root_sectorsR[n])][1] = kept_refsR

    dim = np.size(np.arange(1,root_sectorsL[n,0]+1))-1
    for m in range(0,dim):
        next_sectorL = target_sectorL - np.array([1,0])
        next_sectorR = target_sectorR - np.array([0,1])
        print(next_sectorR)

        next_refsL = sector_refs[perm_key(next_sectorL)]
        next_refsR = sector_refs[perm_key(next_sectorR)]
        kept_refsL = []
        kept_refsR = []
        for k in range(0,np.size(next_refsL,axis=0)):
            if target_sectorL in to_sectors[next_refsL[k]]:
                kept_refsL = np.append(kept_refsL,next_refsL[k])
            if target_sectorR in to_sectors[next_refsR[m]]:
                kept_refsR = np.append(kept_refsR,next_refsR[k])
        hamming_layer_refsL[perm_key(root_sectorsL[n])][m+2] = kept_refsL
        hamming_layer_refsR[perm_key(root_sectorsR[n])][m+2] = kept_refsR

        target_sectorL = next_sectorL
        target_sectorR = next_sectorR

#form basis
hamming_layer_basisL = dict()
hamming_layer_basisR = dict()
for n in range(0,np.size(root_sectorsL,axis=0)):
    hamming_layer_basisL[perm_key(root_sectorsL[n])] = np.zeros((pxp.dim,len(hamming_layer_refsL[perm_key(root_sectorsL[n])])))
    hamming_layer_basisR[perm_key(root_sectorsR[n])] = np.zeros((pxp.dim,len(hamming_layer_refsR[perm_key(root_sectorsR[n])])))
    for m in range(0,len(hamming_layer_refsL[perm_key(root_sectorsL[n])])):
        refsL = hamming_layer_refsL[perm_key(root_sectorsL[n])][m]
        refsR = hamming_layer_refsL[perm_key(root_sectorsL[n])][m]
        tempL = np.zeros(pxp.dim)
        tempR = np.zeros(pxp.dim)
        for k in range(0,np.size(refsL,axis=0)):
            tempL = tempL + ref_state(refsL[k],pxp).prod_basis()
            tempR = tempR + ref_state(refsR[k],pxp).prod_basis()
        tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
        tempR = tempR / np.power(np.vdot(tempR,tempR),0.5)
        hamming_layer_basisL[perm_key(root_sectorsL[n])][:,m] = tempL
        hamming_layer_basisR[perm_key(root_sectorsR[n])][:,m] = tempR
    
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

basis = cube_basisL
for n in range(0,np.size(root_sectorsL,axis=0)):
    basis = np.hstack((basis,hamming_layer_basisL[perm_key(root_sectorsL[n])]))
# if np.size(non_root_basisL)>0:
basis = np.hstack((basis,non_root_basisL))

for n in range(0,np.size(root_sectorsR,axis=0)):
    basis = np.hstack((basis,hamming_layer_basisR[perm_key(root_sectorsR[n])]))
# if np.size(non_root_basisR)>0:
basis = np.hstack((basis,non_root_basisR))

basis = np.hstack((basis,cube_basisR))
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

plt.scatter(e,overlap,marker="x",color="red")
plt.show()
