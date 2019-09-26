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

N = 10
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()

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

#check lie algebra
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

Hz = com(Hp,Hm)
root_check = com(Hz,Hp)

plt.matshow(np.abs(Hz))
plt.show()

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

H = spin_Hamiltonian(pxp,"x")
H.gen()
H.sector.find_eig()
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
plt.matshow(np.abs(H_fsa))
plt.show()

e,u = np.linalg.eigh(H_fsa)
plt.scatter(e,np.log10(np.abs(u[0,:])**2))
plt.show()
u_comp = np.dot(fsa_basis,u)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(exact_overlap,axis=0)):
    max_overlap = 0
    for m in range(0,pxp.dim):
        temp = np.abs(np.vdot(H.sector.eigvectors()[:,m],u_comp[:,n]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.scatter(e,exact_overlap)
plt.show()
