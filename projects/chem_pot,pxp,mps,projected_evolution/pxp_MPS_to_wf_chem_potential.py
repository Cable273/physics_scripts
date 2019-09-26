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

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':36})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N=18
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

#create MPs
def A_up(theta,phi):
    return np.array([[0,1j*np.exp(-1j*phi)],[0,0]])
def A_down(theta,phi):
    return np.array([[np.cos(theta),0],[np.sin(theta),0]])
theta1 = 0.9
theta2 = 2.985
phi1 = 0.188
phi2 = 1.6

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

np.savetxt("z2,entangled_MPS_coef_chem_pot,18",wf)
# e = H.sector.eigvalues()
# u = H.sector.eigvectors()

# psi_energy = np.dot(np.conj(np.transpose(u)),wf)
# z=zm_state(2,1,system)
# neel_energy = np.dot(np.conj(np.transpose(u)),z.prod_basis())

# #entanglement growth
# t=np.arange(0,20,0.1)
# S_mps = np.zeros(np.size(t))
# S_z2 = np.zeros(np.size(t))
# ent = entropy(system)
# pbar=ProgressBar()
# for n in pbar(range(0,np.size(t,axis=0))):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # evolved_state_comp = np.dot(u,evolved_state)

    # evolved_state_neel = time_evolve_state(neel_energy,e,t[n])
    # evolved_state_neel_comp = np.dot(u,evolved_state_neel)

    # S_mps[n] = ent.eval(evolved_state_comp)
    # S_z2[n] = ent.eval(evolved_state_neel_comp)
# plt.plot(t,S_mps,linewidth=2,label="K=2 Entangled MPS")
# plt.plot(t,S_z2,linewidth=2,label="Neel")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$S(t)$")
# plt.title(r"Entropy Growth, $H=PXP+\mu_z \sum_i n_i$, N="+str(system.N)+"\n"+r"$\mu_z=0.325, \theta_1=0.9, \phi_1=0.188, \theta_2=2.985, \phi_2=1.6$")
# plt.legend()
# plt.show()

# eigenvalues = e
# overlap = np.log10(np.abs(psi_energy)**2)
# # overlap = np.log10(np.abs(neel_energy)**2)
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
# plt.title(r"PXP, Eigenstate overlap with K=3 Entangled MPS, N="+str(system.N))
# plt.show()

# t=np.arange(0,20,0.01)
# #MPS fidelity
# f=np.zeros(np.size(t))
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_energy,e,t[n])
    # f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
# plt.plot(t,f,linewidth=2,label="K=2 Entangled MPS")
# for n in range(0,np.size(f,axis=0)):
    # if f[n] < 0.1:
        # cut = n
        # break
# print(np.max(f[cut:]))

# #Neel fidelity
# z=zm_state(2,1,system)
# f = fidelity(z,H).eval(t,z)
# plt.plot(t,f,linewidth=2,label="Z2")

# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# plt.title(r"Fidelity, $H=PXP+\mu_z \sum_i n_i$, N="+str(system.N)+"\n"+r"$\mu_z=0.325, \theta_1=0.9, \phi_1=0.188, \theta_2=2.985, \phi_2=1.6$")
# plt.legend()
# plt.show()
