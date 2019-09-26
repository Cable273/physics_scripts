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
from System_Classes import unlocking_System,U1_system
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

import numpy as np
import scipy as sp
import math

import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def W(N):
    choose = ncr(int(N/2),2)
    return 1/np.power(choose,0.5)*1/np.power(int(N/2)-1,0.5)*N/2

from subcube_functions import perm_key,cube_fsa,find_root_refs

N=16
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

sector_refs = dict()
from_sector = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    bits = pxp.basis[n]
    #find perm sector
    c1 = 0
    c2 = 0
    for m in range(0,np.size(bits,axis=0)):
        if bits[m] == 1:
            if m % 2 == 0:
                c1 = c1 + 1
            else:
                c2 = c2 + 1
    sector = np.array([c1,c2])
    from_sector[pxp.basis_refs[n]] = sector
    key = perm_key(sector,pxp)
    if key in sector_refs.keys():
        sector_refs[key] = np.append(sector_refs[key],pxp.basis_refs[n])
    else:
        sector_refs[key] = [pxp.basis_refs[n]]

consec_flips = dict()
flippable_zeros = dict()
for n in range(0,np.size(pxp.basis_refs,axis=0)):
    bits = pxp.basis[n]
    n_flip = []
    m_flip = []
    for k in range(0,np.size(bits,axis=0)):
        if k == int(np.size(bits)-1):
            kp1 = 0
        else:
            kp1 = k + 1
        if k == 0:
            km1 = int(np.size(bits)-1)
        else:
            km1 = k - 1

        #find number of consecutive flippable bits
        if bits[km1] == 0 and bits[k] == 0 and bits[kp1] == 0:
            if k % 2 == 0:
                n_flip= np.append(n_flip,k)
            else:
                m_flip= np.append(m_flip,k)

    flippable_zeros[pxp.basis_refs[n]] = dict()
    flippable_zeros[pxp.basis_refs[n]][0] = n_flip
    flippable_zeros[pxp.basis_refs[n]][1] = m_flip
    from itertools import combinations
    #no consecutive flips:
    c_flip = 0
    for k in range(1,np.size(n_flip,axis=0)+1):
        combs = np.array(list(combinations(n_flip,k)))
        for u in range(0,np.size(combs,axis=0)):
            temp_bits = np.copy(bits)
            for i in range(0,np.size(combs[u],axis=0)):
                temp_bits[int(combs[u,i])] = 1
            temp_ref = bin_to_int_base_m(temp_bits,pxp.base)
            if temp_ref in pxp.basis_refs:
                c_flip = c_flip + 1
                break

    b_flip = 0
    for k in range(1,np.size(m_flip,axis=0)+1):
        combs = np.array(list(combinations(m_flip,k)))
        for u in range(0,np.size(combs,axis=0)):
            temp_bits = np.copy(bits)
            for i in range(0,np.size(combs[u],axis=0)):
                temp_bits[int(combs[u,i])] = 1
            temp_ref = bin_to_int_base_m(temp_bits,pxp.base)
            if temp_ref in pxp.basis_refs:
                b_flip = b_flip + 1
                break
    consec_flips[pxp.basis_refs[n]] = np.array([c_flip,b_flip])

def fsa_ops_from_root(ref,LR):
    bits = pxp.basis[pxp.keys[ref]]
    one_loc = []
    for m in range(0,np.size(bits,axis=0)):
        if bits[m] == 1:
            one_loc = np.append(one_loc,m)
    if LR == "Left":
        zero_loc = flippable_zeros[ref][1]
    elif LR == "Right":
        zero_loc = flippable_zeros[ref][0]
    print(bits,one_loc+np.ones(np.size(one_loc)),zero_loc+np.ones(np.size(zero_loc)))

    Hp = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        state_bits = np.copy(pxp.basis[n])
        for m in range(0,np.size(state_bits,axis=0)):
            if m in one_loc: 
                if state_bits[m] == 1:
                    new_bits = np.copy(state_bits)
                    new_bits[m] = 0
                    temp_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if temp_ref in pxp.basis_refs:
                        temp_index = pxp.keys[temp_ref]
                        Hp[n,temp_index] = 1
            if m in zero_loc:
                if state_bits[m] == 0:
                    new_bits = np.copy(state_bits)
                    new_bits[m] = 1
                    temp_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if temp_ref in pxp.basis_refs:
                        temp_index = pxp.keys[temp_ref]
                        Hp[n,temp_index] = 1
    Hm = np.conj(np.transpose(Hp))
    return Hp,Hm

def subcube_basis(root_refs,LR,cube_dim,equal_supp=False):
    distinct_subcube_basis = dict()
    for n in range(0,np.size(root_refs,axis=0)):
        distinct_subcube_basis[n] = ref_state(root_refs[n],pxp).prod_basis()
        current_state = distinct_subcube_basis[n]
        Hm,Hp = fsa_ops_from_root(root_refs[n],LR)
        for m in range(0,cube_dim):
            next_state = np.dot(Hp,current_state)
            if (np.abs(next_state)<1e-5).all() == False:
                # next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
                distinct_subcube_basis[n] = np.vstack((distinct_subcube_basis[n],next_state))
            current_state = next_state
        distinct_subcube_basis[n] = np.transpose(distinct_subcube_basis[n])

    #now combine subcube basis
    subcube_basis = distinct_subcube_basis[0]
    for n in range(1,len(distinct_subcube_basis)):
        subcube_basis = subcube_basis + distinct_subcube_basis[n]

    # delete any zeros
    to_del = []
    for n in range(0,np.size(subcube_basis,axis=1)):
        if (np.abs(subcube_basis[:,n])<1e-5).all():
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        subcube_basis = np.delete(subcube_basis,to_del[n],axis=1)

     #if equal supp set all entries to 1
    if equal_supp == True:
        for n in range(0,np.size(subcube_basis,axis=0)):
            for m in range(0,np.size(subcube_basis,axis=1)):
                if np.abs(subcube_basis[n,m])>1e-5:
                    subcube_basis[n,m] = 1
         
    # for n in range(0,np.size(subcube_basis,axis=1)):
        # subcube_basis[:,n] = subcube_basis[:,n] / np.power(np.vdot(subcube_basis[:,n],subcube_basis[:,n]),0.5)

    return subcube_basis

def power(H,n,e,u):
    diag = np.power(e,n)
    return np.dot(u,np.dot(np.diag(diag),np.conj(np.transpose(u))))

def maps_to(ref,target_sector):
    sector = from_sector[ref]
    diff = np.sum(np.abs(target_sector-sector))
    H_power = power(H.sector.matrix(),diff,H.sector.eigvalues(),H.sector.eigvectors())
    index = pxp.keys[ref]
    row = H_power[index,:]
    non_zero = []
    for n in range(0,np.size(row,axis=0)):
        if np.abs(row[n])>1e-5:
            non_zero = np.append(non_zero,n)
    return non_zero

def non_zero_ref_in_basis(basis):
    refs_non_zeros = []
    for n in range(0,np.size(basis,axis=1)):
    # indices = np.array((0,np.size(basis,axis=1)-1)).astype(int)
    # for n in range(0,np.size(indices,axis=0)):
        # psi = basis[:,indices[n]]
        psi = basis[:,n]
        for m in range(0,np.size(psi,axis=0)):
            if np.abs(psi[m])>1e-5:
                refs_non_zeros = np.append(refs_non_zeros,pxp.basis_refs[m])
    refs_non_zeros = np.unique(np.sort(refs_non_zeros))
    return refs_non_zeros

#PXP+PPXP
from Hamiltonian_Classes import H_operations
H0 = spin_Hamiltonian(pxp,"x")
V = Hamiltonian(pxp)
V.site_ops[1] = np.array([[0,1/2],[1/2,0]])
V.site_ops[2] = np.array([[-1/2,0],[0,1/2]])
V.model = np.array([[2,0,1,0],[0,1,0,2]])
V.model_coef = np.array([1,1])
H0.gen()
V.gen()
H = H_operations.add(H0,V,np.array([1,-0.02]))
H.sector.find_eig()

H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()

# N=10
# root_refs = find_root_refs(np.array([3,0]),np.array([3,1]),H,sector_refs,from_sector,pxp,[])
# basis = subcube_basis(root_refs,"Left",4)
# refs_found = non_zero_ref_in_basis(basis)

# print("\n")
# root_refs = find_root_refs(np.array([0,3]),np.array([1,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Right",4)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))

# print("\n")
# root_refs = find_root_refs(np.array([2,0]),np.array([2,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",4)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))

# # #N=12
# root_refs = find_root_refs(np.array([4,0]),np.array([4,1]),H,sector_refs,from_sector,pxp,[])
# basis = subcube_basis(root_refs,"Left",5)
# refs_found = non_zero_ref_in_basis(basis)

# print("\n")
# root_refs = find_root_refs(np.array([3,0]),np.array([3,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",5)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))

# print("\n")
# root_refs = find_root_refs(np.array([2,0]),np.array([2,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",5)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))

# print("\n")
# root_refs = find_root_refs(np.array([2,0]),np.array([2,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",4)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
    
# print("\n")
# root_refs = find_root_refs(np.array([1,0]),np.array([1,4]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",5)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))

# N=14
# print(5,1)
# root_refs = find_root_refs(np.array([5,0]),np.array([5,1]),H,sector_refs,from_sector,pxp,[])
# # basis = subcube_basis(root_refs,"Left",6,equal_supp=True)
# basis = subcube_basis(root_refs,"Left",6,equal_supp=False)
# print(np.size(root_refs))
# refs_found = non_zero_ref_in_basis(basis)

# print("\n")
# print(4,2)
# root_refs = find_root_refs(np.array([4,0]),np.array([4,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # # temp = subcube_basis(root_refs,"Left",6,equal_supp=True)
    # temp = subcube_basis(root_refs,"Left",6,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(3,3)
# root_refs = find_root_refs(np.array([3,0]),np.array([3,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # # temp = subcube_basis(root_refs,"Left",6,equal_supp=True)
    # temp = subcube_basis(root_refs,"Left",6,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(3,2)
# root_refs = find_root_refs(np.array([3,0]),np.array([3,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # # temp = subcube_basis(root_refs,"Left",5,equal_supp=True)
    # temp = subcube_basis(root_refs,"Left",5,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(2,4)
# root_refs = find_root_refs(np.array([2,0]),np.array([2,4]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # # temp = subcube_basis(root_refs,"Left",6,equal_supp=True)
    # temp = subcube_basis(root_refs,"Left",6,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(2,3)
# root_refs = find_root_refs(np.array([2,0]),np.array([2,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # # temp = subcube_basis(root_refs,"Left",5,equal_supp=True)
    # temp = subcube_basis(root_refs,"Left",5,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(1,5)
# root_refs = find_root_refs(np.array([1,0]),np.array([1,5]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",6,equal_supp=False)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# #N=16
print(6,1)
root_refs = find_root_refs(np.array([6,0]),np.array([6,1]),H,sector_refs,from_sector,pxp,[])
basis = subcube_basis(root_refs,"Left",7)
refs_found = non_zero_ref_in_basis(basis)
print(np.size(root_refs))

print("\n")
print(5,2)
root_refs = find_root_refs(np.array([5,0]),np.array([5,2]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",7)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(4,3)
root_refs = find_root_refs(np.array([4,0]),np.array([4,3]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",7)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(4,2)
root_refs = find_root_refs(np.array([4,0]),np.array([4,2]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",6)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(3,4)
root_refs = find_root_refs(np.array([3,0]),np.array([3,4]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",7)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(3,3)
root_refs = find_root_refs(np.array([3,0]),np.array([3,3]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",6)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(2,5)
root_refs = find_root_refs(np.array([2,0]),np.array([2,5]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",7)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(2,4)
root_refs = find_root_refs(np.array([2,0]),np.array([2,4]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",6)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

print("\n")
print(1,6)
root_refs = find_root_refs(np.array([1,0]),np.array([1,6]),H,sector_refs,from_sector,pxp,refs_found)
if np.size(root_refs)>0:
    temp = subcube_basis(root_refs,"Left",7)
    refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    basis = np.hstack((basis,temp))
print(np.size(root_refs))

#N=20
# print(8,1)
# root_refs = find_root_refs(np.array([8,0]),np.array([8,1]),H,sector_refs,from_sector,pxp,[])
# basis = subcube_basis(root_refs,"Left",9)
# refs_found = non_zero_ref_in_basis(basis)
# print(np.size(root_refs))

# print("\n")
# print(7,2)
# root_refs = find_root_refs(np.array([7,0]),np.array([7,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(6,3)
# root_refs = find_root_refs(np.array([6,0]),np.array([6,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(6,2)
# root_refs = find_root_refs(np.array([6,0]),np.array([6,2]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",8)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(5,4)
# root_refs = find_root_refs(np.array([5,0]),np.array([5,4]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(5,3)
# root_refs = find_root_refs(np.array([5,0]),np.array([5,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",8)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(4,5)
# root_refs = find_root_refs(np.array([4,0]),np.array([4,5]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(4,4)
# root_refs = find_root_refs(np.array([4,0]),np.array([4,4]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",8)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(4,3)
# root_refs = find_root_refs(np.array([4,0]),np.array([4,3]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",7)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(3,6)
# root_refs = find_root_refs(np.array([3,0]),np.array([3,6]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(3,5)
# root_refs = find_root_refs(np.array([3,0]),np.array([3,5]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",8)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(3,4)
# root_refs = find_root_refs(np.array([3,0]),np.array([3,4]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",7)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(2,7)
# root_refs = find_root_refs(np.array([2,0]),np.array([2,7]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))
# print("\n")

# print(2,6)
# root_refs = find_root_refs(np.array([2,0]),np.array([2,6]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",8)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

# print("\n")
# print(1,8)
# root_refs = find_root_refs(np.array([1,0]),np.array([1,8]),H,sector_refs,from_sector,pxp,refs_found)
# if np.size(root_refs)>0:
    # temp = subcube_basis(root_refs,"Left",9)
    # refs_found = np.sort(np.unique(np.append(refs_found,non_zero_ref_in_basis(temp))))
    # basis = np.hstack((basis,temp))
# print(np.size(root_refs))

#neel cubes
L_cube_sectors = np.zeros((int(pxp.N/2)+1,2))
R_cube_sectors = np.zeros((int(pxp.N/2)+1,2))
for n in range(0,int(pxp.N/2)+1):
    L_cube_sectors[n,0] = int(pxp.N/2)-n
    R_cube_sectors[n,1] = int(pxp.N/2)-n

L_cube_basis = ref_state(sector_refs[perm_key(L_cube_sectors[0],pxp)][0],pxp).prod_basis()
R_cube_basis = ref_state(sector_refs[perm_key(R_cube_sectors[0],pxp)][0],pxp).prod_basis()
for n in range(1,np.size(L_cube_sectors,axis=0)):
    refsL = sector_refs[perm_key(L_cube_sectors[n],pxp)]
    refsR = sector_refs[perm_key(R_cube_sectors[n],pxp)]

    tempL = np.zeros(pxp.dim)
    tempR = np.zeros(pxp.dim)
    for m in range(0,np.size(refsL,axis=0)):
        tempL[pxp.keys[refsL[m]]] = 1
        tempR[pxp.keys[refsR[m]]] = 1
    # tempL = tempL / np.power(np.vdot(tempL,tempL),0.5)
    # tempR = tempR / np.power(np.vdot(tempR,tempR),0.5)
    L_cube_basis = np.vstack((L_cube_basis,tempL))
    R_cube_basis = np.vstack((R_cube_basis,tempR))
L_cube_basis = np.transpose(L_cube_basis)
R_cube_basis = np.transpose(R_cube_basis)
    
# combine basis, get projected approximations
# basis = fsa_basis
# basis = np.hstack((L_cube_basis,basis))
# basis = np.hstack((basis,R_cube_basis))

# basis = np.hstack((L_cube_basis,basis))
# basis = np.hstack((basis,R_cube_basis))

basis = np.hstack((L_cube_basis,basis))
basis = np.hstack((basis,R_cube_basis))

basis = np.unique(basis,axis=1)
# basis = np.hstack((basis,L_cube_basis))
# basis = np.hstack((basis,np.flip(R_cube_basis,axis=1)))
    
#gram schmidt, QR ruins particle hole
from Calculations import gram_schmidt
gs = gram_schmidt(basis)
gs.ortho()
basis = gs.ortho_basis

#fsa basis for comp
z=zm_state(2,1,pxp)
Hp = np.zeros((pxp.dim,pxp.dim))
for n in range(0,np.size(pxp.basis_refs)):
    state_bits = pxp.basis[n]
    for m in range(0,np.size(state_bits,axis=0)):
        if m % 2 == 0: 
            if state_bits[m] == 1:
                new_bits = np.copy(state_bits)
                new_bits[m] = 0
                temp_ref = bin_to_int_base_m(new_bits,pxp.base)
                if temp_ref in pxp.basis_refs:
                    temp_index = pxp.keys[temp_ref]
                    Hp[n,temp_index] = 1
        else:
            if state_bits[m] == 0:
                new_bits = np.copy(state_bits)
                new_bits[m] = 1
                temp_ref = bin_to_int_base_m(new_bits,pxp.base)
                if temp_ref in pxp.basis_refs:
                    temp_index = pxp.keys[temp_ref]
                    Hp[n,temp_index] = 1
Hm = np.conj(np.transpose(Hp))
fsa_dim = pxp.N
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,fsa_dim):
    next_state = np.dot(Hm,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)

# # #perm approx for comp, FSA
perm_basis = np.zeros(pxp.dim)
keys = list(sector_refs.keys())
for n in range(0,len(sector_refs)):
    if np.size(sector_refs[keys[n]])>1e-5:
        temp_state = np.zeros(pxp.dim)
        for m in range(0,np.size(sector_refs[keys[n]],axis=0)):
            temp_state[pxp.keys[sector_refs[keys[n]][m]]] = 1
        temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
        perm_basis = np.vstack((perm_basis,temp_state))
perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))
perm_basis = np.unique(perm_basis,axis=1)

H_cube = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
H_perm = np.dot(np.transpose(np.conj(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
H_fsa = np.dot(np.transpose(np.conj(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))

e_cube,u_cube = np.linalg.eigh(H_cube)
e_perm,u_perm = np.linalg.eigh(H_perm)
e_fsa,u_fsa = np.linalg.eigh(H_fsa)

overlap_cube = np.log10(np.abs(u_cube[0,:])**2)
overlap_perm = np.log10(np.abs(u_perm[0,:])**2)
overlap_fsa = np.log10(np.abs(u_fsa[0,:])**2)

# np.save("subcube,e,20",e_cube)
# np.save("subcube,overlap,20",overlap_cube)
# np.save("subcub_H,20",H_cube)
# np.save("subcube,basis,20",basis)

# H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
mask = overlap_cube > -5
plt.scatter(e_cube[mask],overlap_cube[mask],marker="s",s=200,alpha=0.6,color="red",label="subcube")
# mask = overlap_perm > -5
# plt.scatter(e_perm[mask],overlap_perm[mask],marker="x",s=200,alpha=0.6,color="blue",label="perm")
# mask = overlap_fsa > -5
# plt.scatter(e_fsa[mask],overlap_fsa[mask],marker="D",s=200,alpha=0.6,color="green",label="fsa")

# plt.title(r"Scar approximations, overlap with Neel State, $PXP+\lambda (PPXP+PXPP)$ N=16")
plt.title(r"$PXP$, Neel Overlap, N="+str(pxp.N))
# plt.title(r"$PXP+\lambda (ZPXP + PXPZ)$, Neel Overlap, N="+str(pxp.N))
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
plt.legend()
plt.show()

def ED_overlap(u_comp_basis):
    exact_overlap = np.zeros(np.size(u_comp_basis,axis=1))
    for n in range(0,np.size(u_comp_basis,axis=1)):
        max_overlap = 0
        for m in range(0,np.size(H.sector.eigvalues(),axis=0)):
            temp = np.abs(np.vdot(u_comp_basis[:,n],H.sector.eigvectors()[:,m]))**2
            if temp > max_overlap:
                max_overlap = temp
        exact_overlap[n] = max_overlap
    return exact_overlap

u_comp_basis_cube = np.dot(basis,u_cube)
u_comp_basis_perm = np.dot(perm_basis,u_perm)
u_comp_basis_fsa = np.dot(fsa_basis,u_fsa)

exact_overlap_cube = ED_overlap(u_comp_basis_cube)
exact_overlap_perm = ED_overlap(u_comp_basis_perm)
exact_overlap_fsa = ED_overlap(u_comp_basis_fsa)
print(exact_overlap_cube)

plt.scatter(e_cube,exact_overlap_cube,marker="s",color="red",s=100,alpha=0.6,label="Subcube")
# plt.scatter(e_perm,exact_overlap_perm,marker="x",s=100,label="Perm")
# plt.scatter(e_fsa,exact_overlap_fsa,label="FSA",marker="D",s=100,color="green")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
# plt.title(r"Scar approximations, overlap with exact eigenstates, $PXP+\lambda (PPXP+PXPP)$ N=16")
plt.title(r"$PXP$, exact overlap N="+str(pxp.N))
# plt.title(r"$PXP+\lambda (ZPXP+PXPZ)$, exact overlap N="+str(pxp.N))
plt.legend()
plt.show()

plt.matshow(np.abs(H_cube))
plt.show()

print("\n")
print("subcube dim="+str(np.size(basis,axis=1)))
print("perm dim="+str(np.size(perm_basis,axis=1)))
print("fsa dim="+str(np.size(fsa_basis,axis=1)))
