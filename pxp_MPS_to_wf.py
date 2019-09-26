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

N=15
D=2
d=2

def A_up(theta,phi):
    return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])

def A_down(theta,phi):
    return np.array([[np.cos(theta),0],[np.sin(theta),0]])

#create MPs
theta3 = 0.1923*math.pi
theta2 = 0
theta1 = 0.8356*math.pi

A_ups = dict()
A_downs = dict()
A_ups[0] = A_up(theta1,0)
A_ups[1] = A_up(theta2,0)
A_ups[2] = A_up(theta3,0)

A_downs[0] = A_down(theta1,0)
A_downs[1] = A_down(theta2,0)
A_downs[2] = A_down(theta3,0)

tensors = dict()
K = 3
for n in range(0,K):
    tensors[n] = np.zeros((2,np.size(A_ups[0],axis=0),np.size(A_ups[0],axis=1)),dtype=complex)
tensors[0][0] = A_downs[0]
tensors[0][1] = A_ups[0]

tensors[1][0] = A_downs[1]
tensors[1][1] = A_ups[1]

tensors[2][0] = A_downs[2]
tensors[2][1] = A_ups[2]

from MPS import periodic_MPS
psi = periodic_MPS(N)
for n in range(0,N,1):
    psi.set_entry(n,tensors[int(n%3)],"both")

#convert MPS -> wf array
system = unlocking_System([0],"periodic",d,N)
system.gen_basis()
wf = np.zeros(system.dim,dtype=complex)
for n in range(0,np.size(system.basis_refs,axis=0)):
    bits = system.basis[n]
    coef = psi.node[0].tensor[bits[0]]
    for m in range(1,np.size(bits,axis=0)):
        coef = np.dot(coef,psi.node[m].tensor[bits[m]])
    coef = np.trace(coef)
    wf[n] = coef
np.save("z3,entangled_MPS_coef,"+str(system.N),wf)
# from Diagnostics import print_wf
# print_wf(wf,system,5e-2)
# print("\n")

# #project MPS wf array to perm basis
# #create perm basis
# # def perm_key(sector):
    # # return bin_to_int_base_m(sector,int(system.N/2+1))

# # sector_refs = dict()
# # sector_key_map = dict()
# # for n in range(0,np.size(system.basis_refs,axis=0)):
    # # bits = system.basis[n]
    # # c1 = 0
    # # c2 = 0
    # # for m in range(0,np.size(bits,axis=0)):
        # # if bits[m] == 1:
            # # if m % 2 == 0 :
                # # c1 = c1 + 1
            # # else:
                # # c2 = c2 + 1
    # # sector = np.array([c1,c2])
    # # if perm_key(sector) in sector_refs.keys():
        # # sector_refs[perm_key(sector)] = np.append(sector_refs[perm_key(sector)],system.basis_refs[n])
    # # else:
        # # sector_refs[perm_key(sector)] = [system.basis_refs[n]]
    # # sector_key_map[perm_key(sector)] = sector

# # perm_basis = np.zeros(system.dim)
# # perm_basis_labels = dict()
# # keys = list(sector_refs.keys())
# # c=0
# # for n in range(0,np.size(keys,axis=0)):
    # # refs = sector_refs[keys[n]]
    # # temp_state = np.zeros(system.dim)
    # # for m in range(0,np.size(refs,axis=0)):
        # # temp_state[system.keys[refs[m]]] = 1
    # # temp_state = temp_state / np.power(np.vdot(temp_state,temp_state),0.5)
    # # perm_basis = np.vstack((perm_basis,temp_state))
    # # perm_basis_labels[c] = sector_key_map[keys[n]]
    # # c = c +1
# # perm_basis = np.transpose(np.delete(perm_basis,0,axis=0))

# # #rot wf to perm basis
# # perm_support = np.zeros((int(system.N/2),int(system.N/2)))
# # wf_rot = np.dot(np.conj(np.transpose(perm_basis)),wf)
# # # wf_rot = wf_rot / np.power(np.vdot(wf_rot,wf_rot),0.5)
# # for n in range(0,np.size(wf_rot,axis=0)):
    # # if np.abs(wf_rot[n])>1e-2:
        # # print(np.abs(wf_rot[n]),perm_basis_labels[n])
        # # perm_support[perm_basis_labels[n][0],perm_basis_labels[n][1]] = wf_rot[n]
# # plt.matshow(np.abs(perm_support))
# # plt.xlabel(r"n")
# # plt.ylabel(r"m",rotation=0)
# # plt.title(r"K=3 Entangled MPS Support on $\vert n,m \rangle$ Perm Sector")
# # plt.show()
    
# #dynamics + fidelity
# H = spin_Hamiltonian(system,"x")
# H.gen()
# H.sector.find_eig()
# e = H.sector.eigvalues()
# u = H.sector.eigvectors()
# psi_energy = np.dot(np.conj(np.transpose(u)),wf)
# eigenvalues = e
# overlap = np.log10(np.abs(psi_energy)**2)
# to_del=[]
# for n in range(0,np.size(overlap,axis=0)):
    # if overlap[n] <-10:
        # to_del = np.append(to_del,n)
# for n in range(np.size(to_del,axis=0)-1,-1,-1):
    # overlap=np.delete(overlap,to_del[n])
    # eigenvalues=np.delete(eigenvalues,to_del[n])
    
# plt.scatter(eigenvalues,overlap)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# plt.title(r"PXP, Eigenstate overlap with K=3 Entangled MPS, N=18")
# plt.show()

# t=np.arange(0,20,0.01)
# f=np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(t) \vert \psi(0) \rangle \vert^2$")
# plt.title(r"PXP, K=3 Entangled MPS Fidelity, N=18")
# plt.plot(t,f,label="K=3 Entangled MPS")
# z3=zm_state(3,1,system)
# f = fidelity(z3,H).eval(t,z3)
# plt.plot(t,f,label="Z3")
# plt.legend()
# plt.show()


