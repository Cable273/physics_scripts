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

def L(J,M,pm):
    if pm == "p":
        return np.power(J*(J+1)-M*(M+1),0.5)
    else:
        return np.power(J*(J+1)-M*(M-1),0.5)

def m_pair_hash(m_pair,s1,s2):
    base = int(np.max(np.array((4*s1,4*s2))))
    temp = np.copy(m_pair)
    temp[0] = temp[0] + s1
    temp[1] = temp[1] + s2
    temp = 2*temp
    key = bin_to_int_base_m(temp,base)
    return key

s1 = 0.5
s2 = 0.5

J=1
m1 = np.arange(-s1,s1+1)
m2 = np.arange(-s2,s2+1)

c=0
m_pairs = np.zeros((np.size(m1)*np.size(m2),2))
for n in range(0,np.size(m1,axis=0)):
    for m in range(0,np.size(m2,axis=0)):
        m_pairs[c] = np.array((m1[n],m2[m]))
        c = c+1

m_refs = np.zeros(np.size(m_pairs,axis=0),dtype=int)
m_key = dict()
for n in range(0,np.size(m_pairs,axis=0)):
    m_refs[n] = m_pair_hash(m_pairs[n],s1,s2)
    m_key[m_refs[n]] = n

#check have a unique key for every pair
if np.size(m_refs) != np.size(np.unique(m_refs)):
    print("ERROR: Refs for spin pairs (m1,m2) not unique!")

print(m_pairs)

M=np.zeros((np.size(m_pairs,axis=0),np.size(m_pairs,axis=0)))
for n in range(0,np.size(m_pairs,axis=0)):
    m=np.copy(m_pairs[n])

    if m[0] -1 in m1:
        ref = m_pair_hash(np.array((m[0]-1,m[1])),s1,s2)
        index = m_key[ref]
        M[n,index] = L(s1,m[0]-1,"p")

    if m[1]-1 in m2:
        ref = m_pair_hash(np.array((m[0],m[1]-1)),s1,s2)
        index = m_key[ref]
        M[n,index] = L(s2,m[1]-1,"p")

print(M)
e,u = np.linalg.eigh(M)
print(m_pairs)
print(u)
