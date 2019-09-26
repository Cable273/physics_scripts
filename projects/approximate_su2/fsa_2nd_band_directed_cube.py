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

N = 12
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
            non_zero = np.append(non_zero,n)
    return non_zero
        


H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()

# refs = sector_refs[perm_key(np.array([3,0]),pxp)]
# for n in range(0,np.size(refs,axis=0)):
    # print(maps_to(refs[n],[0,3]))
# # psi = bin_state(np.array([]))
# # print(maps_to(0,[3,0]))

# # N=10
# root_refs = find_root_refs(np.array([3,0]),np.array([3,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",4)

# print("\n")
# root_refs = find_root_refs(np.array([0,3]),np.array([1,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",4)))

# print("\n")
# root_refs = find_root_refs(np.array([2,1]),np.array([2,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",4)))

# print("\n")
# root_refs = find_root_refs(np.array([1,2]),np.array([2,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",4)))

# #N=12
root_refs = find_root_refs(np.array([4,0]),np.array([4,1]),H,sector_refs,from_sector,pxp)
basis = subcube_basis(root_refs,"Left",5)

print("\n")
root_refs = find_root_refs(np.array([3,0]),np.array([3,2]),H,sector_refs,from_sector,pxp)
basis = np.hstack((basis,subcube_basis(root_refs,"Left",5)))

print("\n")
root_refs = find_root_refs(np.array([0,3]),np.array([2,3]),H,sector_refs,from_sector,pxp)
basis = np.hstack((basis,subcube_basis(root_refs,"Right",5)))

print("\n")
root_refs = find_root_refs(np.array([2,0]),np.array([2,2]),H,sector_refs,from_sector,pxp)
basis = np.hstack((basis,subcube_basis(root_refs,"Left",4)))

print("\n")
root_refs = find_root_refs(np.array([0,4]),np.array([1,4]),H,sector_refs,from_sector,pxp)
basis = np.hstack((basis,subcube_basis(root_refs,"Right",5)))

#N=14
# root_refs = find_root_refs(np.array([5,0]),np.array([5,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",6)

# print("\n")
# root_refs = find_root_refs(np.array([4,0]),np.array([4,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([3,0]),np.array([3,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([3,0]),np.array([3,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",5)))

# print("\n")
# root_refs = find_root_refs(np.array([0,4]),np.array([2,4]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",6)))

# print("\n")
# root_refs = find_root_refs(np.array([0,3]),np.array([2,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",5)))

# print("\n")
# root_refs = find_root_refs(np.array([0,5]),np.array([1,5]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",6)))

#N=16
# root_refs = find_root_refs(np.array([6,0]),np.array([6,1]),H,sector_refs,from_sector,pxp)
# basis = subcube_basis(root_refs,"Left",7)

# print("\n")
# root_refs = find_root_refs(np.array([5,0]),np.array([5,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",7)))

# print("\n")
# root_refs = find_root_refs(np.array([4,0]),np.array([4,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",7)))

# print("\n")
# root_refs = find_root_refs(np.array([4,0]),np.array([4,2]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([0,4]),np.array([3,4]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",7)))

# print("\n")
# root_refs = find_root_refs(np.array([3,0]),np.array([3,3]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([0,5]),np.array([2,5]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",7)))

# print("\n")
# root_refs = find_root_refs(np.array([2,0]),np.array([2,4]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Left",6)))

# print("\n")
# root_refs = find_root_refs(np.array([0,6]),np.array([1,6]),H,sector_refs,from_sector,pxp)
# basis = np.hstack((basis,subcube_basis(root_refs,"Right",7)))


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
print(np.shape(basis))
# basis = fsa_basis
print("FSA+CUBE FSA dim="+str(np.size(basis,axis=1)))

# from Diagnostics import print_wf
# for n in range(0,np.size(basis,axis=1)):
    # print("\n")
    # print_wf(basis[:,n],pxp,1e-2)

# H_fsa = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
# # H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# plt.matshow(np.abs(H_fsa))
# plt.show()

# z=zm_state(2,1,pxp)
# eig_overlap(z,H).plot()

# e,u = np.linalg.eigh(H_fsa)
# # index = int(np.size(u,axis=0)-1)
# index = 0
# eigenvalues = e
# to_del=[]
# overlap = np.log10(np.abs(u[index,:]**2))
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] < - 5:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
    
# plt.scatter(eigenvalues,overlap,marker="x",color="red",s=100)
# plt.show()

# u_comp = np.dot(basis,u)
# exact_overlap = np.zeros(np.size(e))
# for n in range(0,np.size(e,axis=0)):
    # max_overlap = 0
    # for m in range(0,pxp.dim):
        # temp = np.abs(np.vdot(H.sector.eigvectors()[:,m],u_comp[:,n]))**2
        # if temp > max_overlap:
            # max_overlap = temp
    # exact_overlap[n] = max_overlap
# print(exact_overlap)
# plt.scatter(e,exact_overlap)
# plt.show()


from Diagnostics import print_wf
for m in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,m],pxp,1e-2)

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
e,u = np.linalg.eigh(H_rot)
eigenvalues = e
overlap = np.log10(np.abs(u[0,:])**2)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])

H.sector.find_eig()
z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
plt.scatter(eigenvalues,overlap,marker="s",s=200,alpha=0.6,color="red",label="subcube")

#perm approx for comp, FSA
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

H_perm = np.dot(np.transpose(np.conj(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
ep,up = np.linalg.eigh(H_perm)
eigenvalues = ep
overlap_perm = np.log10(np.abs(up[0,:])**2)
to_del=[]
for n in range(0,np.size(overlap_perm,axis=0)):
    if overlap_perm[n] < - 5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap_perm=np.delete(overlap_perm,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
    
plt.scatter(eigenvalues,overlap_perm,marker="x",s=100,label="Perm")

H_fsa = np.dot(np.transpose(np.conj(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
ef,uf = np.linalg.eigh(H_fsa)
overlap_fsa = np.log10(np.abs(uf[0,:])**2)
plt.scatter(ef,overlap_fsa,marker="D",s=100,color="green",alpha =0.6,label="FSA")
plt.title(r"Scar approximations, overlap with Neel State, PXP N=16")
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
plt.scatter(e,exact_overlap,marker="s",color="red",s=100,alpha=0.6,label="Subcube")
plt.scatter(ep,exact_overlap_perm,marker="x",s=100,label="Perm")
plt.scatter(ef,exact_overlap_fsa,label="FSA",marker="D",s=100,color="green")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
plt.title(r"Scar approximations, overlap with exact eigenstates, PXP N=16")
plt.legend()
plt.show()

plt.matshow(np.abs(H_rot))
plt.show()
# plt.matshow(np.abs(H_perm))
# plt.show()
# plt.matshow(np.abs(H_fsa))
# plt.show()

print("\n")
print("subcube dim="+str(np.size(basis,axis=1)))
print("perm dim="+str(np.size(perm_basis,axis=1)))
print("fsa dim="+str(np.size(fsa_basis,axis=1)))
 

print(pxp.dim)
