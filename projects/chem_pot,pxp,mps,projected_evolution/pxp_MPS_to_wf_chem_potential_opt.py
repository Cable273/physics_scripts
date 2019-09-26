#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
file_dir = '/localhome/pykb/Python/Tensor_Train/'
sys.path.append(file_dir)
import numpy as np
from MPS import mpo,mps
from Tensor_Train import rail_network
from common_MPOs import common_mpo
from compression import var_compress,svd_compress
import matplotlib.pyplot as plt
from DMRG import *
from progressbar import ProgressBar

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
import math

N=10
D=2
d=2

system = unlocking_System([0],"periodic",d,N)
system.gen_basis()
#dynamics + fidelity
H = Hamiltonian(system)
H.site_ops[1] = np.array([[0,1],[1,0]])
H.site_ops[2] = np.array([[0,0],[0,1]])
H.model = np.array([[0,1,0],[2]])
H.model_coef = np.array([1,0.325])
H.gen()
H.sector.find_eig()
e = H.sector.eigvalues()
u = H.sector.eigvectors()

ent = entropy(system)

def long_time_entropy(phi2,t):
    #create MPs
    def A_up(theta,phi):
        return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])
    def A_down(theta,phi):
        return np.array([[np.cos(theta),0],[np.sin(theta),0]])
    theta1 = 0.9
    theta2 = 2.985
    phi1 = 0.188
    # phi2 = 1.3

    A_ups = dict()
    A_downs = dict()
    A_ups[0] = A_up(theta1,phi1)
    A_ups[1] = A_up(theta2,phi2)

    A_downs[0] = A_down(theta1,phi1)
    A_downs[1] = A_down(theta2,phi2)

    tensors = dict()
    K = 2
    for n in range(0,K):
        tensors[n] = np.zeros((2,np.size(A_ups[0],axis=0),np.size(A_ups[0],axis=1)),dtype=complex)
    tensors[0][0] = A_downs[0]
    tensors[0][1] = A_ups[0]
    tensors[1][0] = A_downs[1]
    tensors[1][1] = A_ups[1]

    from MPS import periodic_MPS
    psi = periodic_MPS(N)
    for n in range(0,N,1):
        psi.set_entry(n,tensors[int(n%2)],"both")

    #convert MPS -> wf array
    wf = np.zeros(system.dim,dtype=complex)
    for n in range(0,np.size(system.basis_refs,axis=0)):
        bits = system.basis[n]
        coef = psi.node[0].tensor[bits[0]]
        for m in range(1,np.size(bits,axis=0)):
            coef = np.dot(coef,psi.node[m].tensor[bits[m]])
        coef = np.trace(coef)
        wf[n] = coef

    psi_energy = np.dot(np.conj(np.transpose(u)),wf)
    evolved_state = time_evolve_state(psi_energy,e,t)
    evolved_state_comp = np.dot(np.conj(np.transpose(u)),evolved_state)
    S = ent.eval(evolved_state_comp)
    print(phi2,S)
    return S
phi2_vals = np.arange(0,2*math.pi+0.01,0.01)
S = np.zeros(np.size(phi2_vals))
for n in range(0,np.size(S,axis=0)):
    S[n] = long_time_entropy(phi2_vals[n],15)
min_index = np.argmin(S)
print(phi2_vals[min_index],S[min_index])
plt.plot(phi2_vals,S)
plt.show()


