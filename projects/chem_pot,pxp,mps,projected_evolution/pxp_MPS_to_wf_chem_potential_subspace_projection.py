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

N=16
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

fsa_basis = np.load("./basis_data/fsa_basis,16.npy")
perm_basis = np.load("./basis_data/perm_basis,16.npy")
subcube_basis = np.load("./basis_data/Subcube_basis,16.npy")
H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
H_perm = np.dot(np.conj(np.transpose(perm_basis)),np.dot(H.sector.matrix(),perm_basis))
H_cube = np.dot(np.conj(np.transpose(subcube_basis)),np.dot(H.sector.matrix(),subcube_basis))
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
e_perm,u_perm = np.linalg.eigh(H_perm)
e_cube,u_cube = np.linalg.eigh(H_cube)

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

e = H.sector.eigvalues()
u = H.sector.eigvectors()

psi_fsa = np.dot(np.conj(np.transpose(fsa_basis)),wf)
psi_perm = np.dot(np.conj(np.transpose(perm_basis)),wf)
psi_cube = np.dot(np.conj(np.transpose(subcube_basis)),wf)
psi_fsa = psi_fsa / np.power(np.vdot(psi_fsa,psi_fsa),0.5)
psi_perm = psi_perm / np.power(np.vdot(psi_perm,psi_perm),0.5)
psi_cube = psi_cube / np.power(np.vdot(psi_cube,psi_cube),0.5)

psi_fsa_energy = np.dot(np.conj(np.transpose(u_fsa)),psi_fsa)
psi_perm_energy = np.dot(np.conj(np.transpose(u_perm)),psi_perm)
psi_cube_energy = np.dot(np.conj(np.transpose(u_cube)),psi_cube)

psi_energy = np.dot(np.conj(np.transpose(u)),wf)

t=np.arange(0,20,0.01)
f_fsa = np.zeros(np.size(t))
f_perm = np.zeros(np.size(t))
f_cube = np.zeros(np.size(t))
f_exact = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    evolved_state_fsa = time_evolve_state(psi_fsa_energy,e_fsa,t[n])
    evolved_state_perm = time_evolve_state(psi_perm_energy,e_perm,t[n])
    evolved_state_cube = time_evolve_state(psi_cube_energy,e_cube,t[n])

    f_exact[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
    f_fsa[n] = np.abs(np.vdot(evolved_state_fsa,psi_fsa_energy))**2
    f_perm[n] = np.abs(np.vdot(evolved_state_perm,psi_perm_energy))**2
    f_cube[n] = np.abs(np.vdot(evolved_state_cube,psi_cube_energy))**2

plt.plot(t,f_exact,linewidth=2,label="Exact")
plt.plot(t,f_fsa,linewidth=2,label="FSA")
plt.plot(t,f_perm,linewidth=2,label="Perm")
plt.plot(t,f_cube,linewidth=2,label="Cube")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"K=2 Entangled MPS Fidelity, $H=PXP + \mu_z \sum_i n_i, N=$"+str(system.N)+"\n"+r"$\mu_z=0.325, \theta_1=0.9, \phi_1=0.188, \theta_2=2.985, \phi_2=1.6$")
plt.show()
