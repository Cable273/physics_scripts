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

N = 14
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

def Hm_from_ref(ref):
    bits = pxp.basis[pxp.keys[ref]]
    one_loc = []
    for m in range(0,np.size(bits,axis=0)):
        if bits[m] == 1:
            one_loc = np.append(one_loc,m)

    Hm = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        state_bits = pxp.basis[n]
        for m in range(0,np.size(one_loc,axis=0)):
            if state_bits[int(one_loc[m])] == 1:
                new_bits = np.copy(state_bits)
                new_bits[int(one_loc[m])] = 0
                temp_ref = bin_to_int_base_m(new_bits,pxp.base)
                Hm[n,pxp.keys[temp_ref]] = 1
    return Hm

def subcube_basis_from_sector(sector):
    root_basis = dict()
    refs = sector_refs[perm_key(sector,pxp)]
    for n in range(0,np.size(refs,axis=0)):
        root_basis[n] = np.zeros(pxp.dim)
        Hm = Hm_from_ref(refs[n])
        Hm = np.conj(np.transpose(Hm))
        current_state = ref_state(refs[n],pxp).prod_basis()
        while (np.abs(current_state)<1e-5).all() == False:
            root_basis[n] = np.vstack((root_basis[n],current_state))
            current_state = np.dot(Hm,current_state)
        root_basis[n] = np.transpose(np.delete(root_basis[n],0,axis=0))

    #form superposition from all roots
    basis = root_basis[0]
    for n in range(1,len(root_basis)):
        basis = basis + root_basis[n]

    #normalize
    for n in range(0,np.size(basis,axis=1)):
        basis[:,n] = basis[:,n] / np.power(np.vdot(basis[:,n],basis[:,n]),0.5)
    return basis
        
basis = subcube_basis_from_sector(np.array([5,1]))
basis = np.hstack((basis,subcube_basis_from_sector(np.array([4,2]))))
basis = np.hstack((basis,subcube_basis_from_sector(np.array([3,3]))))
basis = np.hstack((basis,subcube_basis_from_sector(np.array([2,4]))))
basis = np.hstack((basis,subcube_basis_from_sector(np.array([1,5]))))

H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()

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

from Diagnostics import print_wf
for n in range(0,np.size(basis,axis=1)):
    print("\n")
    print_wf(basis[:,n],pxp,1e-2)

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
        

