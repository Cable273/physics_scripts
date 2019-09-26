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

N = 8
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
            elif k % 2 != 0:
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
    print(bits,one_loc,zero_loc)

    Hp = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        state_bits = pxp.basis[n]
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

def subcube_basis(root_refs,LR,cube_dim):
    distinct_subcube_basis = dict()
    for n in range(0,np.size(root_refs,axis=0)):
        distinct_subcube_basis[n] = ref_state(root_refs[n],pxp).prod_basis()
        current_state = distinct_subcube_basis[n]
        Hm,Hp = fsa_ops_from_root(root_refs[n],LR)
        for m in range(0,cube_dim):
            next_state = np.dot(Hp,current_state)
            if (np.abs(next_state)<1e-5).all() == False:
                next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
                distinct_subcube_basis[n] = np.vstack((distinct_subcube_basis[n],next_state))
            current_state = next_state
        distinct_subcube_basis[n] = np.transpose(distinct_subcube_basis[n])

    #now combine subcube basis
    subcube_basis = distinct_subcube_basis[0]
    for n in range(1,len(distinct_subcube_basis)):
        subcube_basis = subcube_basis + distinct_subcube_basis[n]

    #delete any zeros
    to_del = []
    for n in range(0,np.size(subcube_basis,axis=1)):
        if (np.abs(subcube_basis[:,n])<1e-5).all():
            to_del = np.append(to_del,n)
    for n in range(np.size(to_del,axis=0)-1,-1,-1):
        subcube_basis = np.delete(subcube_basis,to_del[n],axis=1)

    for n in range(0,np.size(subcube_basis,axis=1)):
        subcube_basis[:,n] = subcube_basis[:,n] / np.power(np.vdot(subcube_basis[:,n],subcube_basis[:,n]),0.5)

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
            comes_from = from_sector[pxp.basis_refs[n]]
            if (comes_from == target_sector).all():
                non_zero = np.append(non_zero,n)
    non_zero_refs = np.zeros(np.size(non_zero))
    for n in range(0,np.size(non_zero,axis=0)):
        non_zero_refs[n] = pxp.basis_refs[int(non_zero[n])]
    return non_zero_refs

def fsa_ops_root_to_target(root_ref,target_ref):
    root_bits = pxp.basis[pxp.keys[root_ref]]
    target_bits = pxp.basis[pxp.keys[target_ref]]
    sm_loc = []
    sp_loc = []
    for n in range(0,np.size(root_bits,axis=0)):
        if root_bits[n] == 1 and target_bits[n] == 0:
            sm_loc = np.append(sm_loc,n)
        if root_bits[n] == 0 and target_bits[n] == 1:
            sp_loc = np.append(sp_loc,n)

    #construct FSA op
    Hp = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        state_bits = pxp.basis[n]
        for m in range(0,np.size(state_bits,axis=0)):
            if m == np.size(state_bits)-1:
                mp1 = 0
            else:
                mp1 = m + 1
            if m == 0:
                mm1 = np.size(state_bits)-1
            else:
                mm1 = m - 1

            if m in sm_loc:
                if state_bits[mm1] == 0 and state_bits[m] == 1 and state_bits[mp1] == 0:
                    new_bits = np.copy(state_bits)
                    new_bits[m] = 0
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        new_index = pxp.keys[new_ref]
                        Hp[n,new_index] = 1
            if m in sp_loc:
                if state_bits[mm1] == 0 and state_bits[m] == 0 and state_bits[mp1] == 0:
                    new_bits = np.copy(state_bits)
                    new_bits[m] = 1
                    new_ref = bin_to_int_base_m(new_bits,pxp.base)
                    if new_ref in pxp.basis_refs:
                        new_index = pxp.keys[new_ref]
                        Hp[n,new_index] = 1
    Hm = np.conj(np.transpose(Hp))
    return Hp,Hm

H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()

# refs = sector_refs[perm_key(np.array([2,2]),pxp)]
# for n in range(0,np.size(refs,axis=0)):
    # print(pxp.basis[pxp.keys[refs[n]]])
    

# root_refs = find_root_refs(np.array([3,0]),np.array([3,1]),H,sector_refs,from_sector,pxp)
# for n in range(0,np.size(root_refs,axis=0)):
    # goes_to = maps_to(root_refs[n],np.array([0,1]))
    # print(pxp.basis[pxp.keys[root_refs[n]]],goes_to)
    # for m in range(0,np.size(goes_to,axis=0)):
        # Hp,Hm = fsa_ops_root_to_target(root_refs[n],goes_to[m])
        
    

# refs = sector_refs[perm_key(np.array([3,0]),pxp)]
# for n in range(0,np.size(refs,axis=0)):
    # print(maps_to(refs[n],[0,3]))
# # psi = bin_state(np.array([]))
# # print(maps_to(0,[3,0]))

# N=8
root_refs = find_root_refs(np.array([2,0]),np.array([2,1]),H,sector_refs,from_sector,pxp)
basis = subcube_basis(root_refs,"Left",3)

root_refs = find_root_refs(np.array([0,2]),np.array([1,2]),H,sector_refs,from_sector,pxp)
basis = subcube_basis(root_refs,"Right",3)


# # N=10
# root_refs = find_root_refs(np.array([3,0]),np.array([3,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",4)

# print("\n")
# root_refs = find_root_refs(np.array([0,3]),np.array([1,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",4)))

# print("\n")
# root_refs = find_root_refs(np.array([2,0]),np.array([2,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",4)))


# #N=12
# root_refs = find_root_refs(np.array([4,0]),np.array([4,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",5)


# print("\n")
# root_refs = find_root_refs(np.array([0,4]),np.array([1,4]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",5)))

# print("\n")
# root_refs = find_root_refs(np.array([3,1]),np.array([3,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",5)))

# print("\n")
# root_refs = find_root_refs(np.array([1,3]),np.array([2,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",5)))


#N=14
# root_refs = find_root_refs(np.array([5,0]),np.array([5,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",6)

# print("\n")
# root_refs = find_root_refs(np.array([0,5]),np.array([1,5]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",6)))

# print("\n")
# root_refs = find_root_refs(np.array([4,1]),np.array([4,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([1,4]),np.array([2,4]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",6)))

# print("\n")
# root_refs = find_root_refs(np.array([3,0]),np.array([3,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# include Neel fsa (orig)
#fsa Hamiltonian, H= H+ + H-
Hp1 = Hamiltonian(pxp)
Hp1.site_ops[1] = np.array([[0,1],[0,0]])
Hp1.model = np.array([[1]])
Hp1.model_coef = np.array([1])
Hp1.gen(parity=0)
Hp2 = Hamiltonian(pxp)
Hp2.site_ops[1] = np.array([[0,0],[1,0]])
Hp2.model = np.array([[1]])
Hp2.model_coef = np.array([1])
Hp2.gen(parity=1)
Hm = Hp1.sector.matrix()+Hp2.sector.matrix()
Hp = np.conj(np.transpose(Hm))

#fsa basis from Neel
z=zm_state(2,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
fsa_dim = pxp.N
for n in range(0,fsa_dim):
    new_state = np.dot(Hp,current_state)
    new_state = new_state / np.power(np.vdot(new_state,new_state),0.5)
    fsa_basis = np.vstack((fsa_basis,new_state))
    current_state = new_state
fsa_basis = np.transpose(fsa_basis)

#combine basis, get projected approximations
basis = np.hstack((basis,fsa_basis))
basis = np.unique(basis,axis=1)
basis,temp = np.linalg.qr(basis)
# basis = fsa_basis
print("FSA+CUBE FSA dim="+str(np.size(basis,axis=1)))

# from Diagnostics import print_wf
# for n in range(0,np.size(basis,axis=1)):
    # print("\n")
    # print_wf(basis[:,n],pxp,1e-2)

H_fsa = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
plt.matshow(np.abs(H_fsa))
plt.show()

z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()

e,u = np.linalg.eigh(H_fsa)
# index = int(np.size(u,axis=0)-1)
index = 0
eigenvalues = e
to_del=[]
overlap = np.log10(np.abs(u[index,:]**2))
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] < - 5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
    
plt.scatter(eigenvalues,overlap,marker="x",color="red",s=100)
plt.show()

u_comp = np.dot(basis,u)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(e,axis=0)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(H.sector.eigvectors()[:,m],u_comp[:,n]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
print(exact_overlap)
plt.scatter(e,exact_overlap)
plt.show()
