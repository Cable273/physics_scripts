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
H.sector.find_eig()

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

#find identical refs in perm sector
def group_perm_refs(sector):
    refs = sector_refs[perm_key(sector)]
    sector_labels = []

    uniq_keys = []
    for m in range(0,np.size(refs,axis=0)):
        goes_to = to_sectors[refs[m]]
        #goes to key
        numbers = []
        for k in range(0,np.size(goes_to,axis=0)):
            numbers = np.append(numbers,bin_to_int_base_m(goes_to[k],int(pxp.N/2+1)))
        number_max = (pxp.N/2+1)*(pxp.N/2)+(pxp.N/2)
        key = bin_to_int_base_m(numbers,number_max+1)
        uniq_keys = np.unique(np.append(uniq_keys,key))

    uniq_refs = dict()
    for m in range(0,np.size(uniq_keys,axis=0)):
        uniq_refs[uniq_keys[m]] = []

    for m in range(0,np.size(refs,axis=0)):
        goes_to = to_sectors[refs[m]]
        #goes to key
        numbers = []
        for k in range(0,np.size(goes_to,axis=0)):
            numbers = np.append(numbers,bin_to_int_base_m(goes_to[k],int(pxp.N/2+1)))
        number_max = (pxp.N/2+1)*(pxp.N/2)+(pxp.N/2)
        key = bin_to_int_base_m(numbers,number_max+1)
        uniq_refs[key] = np.append(uniq_refs[key],refs[m])
    return uniq_refs

def root_list_to_basis(roots):
    temp_state = np.zeros(pxp.dim)
    for m in range(0,np.size(roots,axis=0)):
        temp_state[pxp.keys[roots[m]]] = 1
    temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    return temp_state
        
from itertools import product
perm_sectors = np.array(list(product(np.arange(0,pxp.N/2+1),np.arange(0,pxp.N/2+1))))
basis = np.zeros(pxp.dim)
for m in range(0,np.size(perm_sectors,axis=0)):
    uniq_refs = group_perm_refs(perm_sectors[m])
    keys = list(uniq_refs.keys())
    for n in range(0,np.size(keys,axis=0)):
        basis = np.vstack((basis,root_list_to_basis(uniq_refs[keys[n]])))
basis = np.transpose(np.delete(basis,0,axis=0))

from Diagnostics import print_wf
for m in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,m],pxp,1e-2)
        
H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis)) 
e,u = np.linalg.eigh(H_rot)
psi_energy = u[np.size(u,axis=1)-1,:]
# psi_energy = u[0,:]
overlap = np.log10(np.abs(psi_energy)**2)

H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()

plt.scatter(e,overlap,marker="s",alpha=0.6,color="red",s=100,label="Perm Unique")

# perm approx for comp, FSA
perm_basis = np.zeros(pxp.dim)
for n in range(0,len(sector_refs)):
    if np.size(sector_refs[n])>1e-5:
        temp_state = np.zeros(pxp.dim)
        for m in range(0,np.size(sector_refs[n],axis=0)):
            temp_state[pxp.keys[sector_refs[n][m]]] = 1
        temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
        perm_basis = np.vstack((perm_basis,temp_state))
perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))
perm_basis = np.unique(perm_basis,axis=1)

H_perm = np.dot(np.transpose(np.conj(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
ep,up = np.linalg.eigh(H_perm)
overlap_perm = np.log10(np.abs(up[0,:])**2)
plt.scatter(ep,overlap_perm,marker="x",s=100,label="Perm")

# fsa
z=zm_state(2,1,pxp)
fsa_hamming = find_hamming_sectors(z.bits,pxp)
fsa_basis = np.zeros((pxp.dim,len(fsa_hamming)))
for n in range(0,len(fsa_hamming)):
    temp_state = np.zeros(pxp.dim)
    refs = fsa_hamming[n]
    for m in range(0,np.size(refs,axis=0)):
        temp_state[pxp.keys[refs[m]]] = 1
    temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    fsa_basis[:,n] = temp_state

H_fsa = np.dot(np.transpose(np.conj(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
ef,uf = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(uf[0,:])**2)
plt.scatter(ef,overlap_fsa,marker="D",s=100,color="green",alpha =0.6,label="FSA")
plt.legend()
plt.show()

u_comp_basis = np.dot(basis,u)
u_comp_basis_perm = np.dot(perm_basis,up)
u_comp_basis_fsa = np.dot(fsa_basis,uf)

exact_overlap = np.zeros(np.size(u_comp_basis,axis=1))
for n in range(0,np.size(u_comp_basis,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap

exact_overlap_perm = np.zeros(np.size(u_comp_basis_perm,axis=1))
for n in range(0,np.size(u_comp_basis_perm,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis_perm[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_perm[n] = max_overlap

exact_overlap_fsa = np.zeros(np.size(u_comp_basis_fsa,axis=1))
for n in range(0,np.size(u_comp_basis_fsa,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
        temp = np.abs(np.vdot(u_comp_basis_fsa[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap_fsa[n] = max_overlap


print(exact_overlap)
plt.scatter(e,exact_overlap,marker="s",color="red",s=100,alpha=0.6,label="Perm Unique")
plt.scatter(ep,exact_overlap_perm,marker="x",s=100,label="Perm")
plt.scatter(ef,exact_overlap_fsa,label="FSA",marker="D",s=100,color="green")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.legend()
plt.show()


plt.matshow(np.abs(H_rot))
plt.show()
# plt.matshow(np.abs(H_perm))
# plt.show()
# plt.matshow(np.abs(H_fsa))
# plt.show()

# print("\n")
# print("perm unique dim="+str(np.size(basis,axis=1)))
# print("perm dim="+str(np.size(perm_basis,axis=1)))
# print("fsa dim="+str(np.size(fsa_basis,axis=1)))
# print(pxp.dim)

    

