#!/usr/bin/env python# -*- coding: utf-8 -*-

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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
# from Plotting_Classes import eig_overlap,fidelity,entropy,energy_basis
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state, gram_schmidt

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 6
base=3
pxp = unlocking_System([0,1,2],"periodic",base,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

# H_clock = clock_Hamiltonian(pxp,pxp_syms)
H_clock = Hamiltonian(pxp,pxp_syms)
H_clock.site_ops[1] = np.array([[0,1j,-1j],[-1j,0,1j],[1j,-1j,0]])
H_clock.model = np.array([[0,1,0]])
H_clock.model_coef = np.array([1])
H_clock.gen()
H_clock.sector.find_eig()

# H_spin = spin_Hamiltonian(pxp,"x",pxp_syms)
H_spin = Hamiltonian(pxp,pxp_syms)
H_spin.site_ops[1] = np.array([[0,1,0],[1,0,1],[0,1,0]])
H_spin.model = np.array([[0,1,0]])
H_spin.model_coef = np.array([1])
H_spin.gen()
H_spin.sector.find_eig()

C=H_clock.site_ops[1]
X=H_spin.site_ops[1]
ec,uc = np.linalg.eigh(C)
es,us = np.linalg.eigh(X)

a = np.abs(es[0]/ec[0])
C=a*C
ec,uc = np.linalg.eigh(C)

ucx = np.dot(uc,np.conj(np.transpose(us)))
uxc = np.dot(us,np.conj(np.transpose(uc)))

P0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
z=zm_state(2,1,pxp)

P0_spin_basis = np.dot(np.conj(np.transpose(ucx)),np.dot(P0,ucx))
print(P0_spin_basis)
# X = np.dot(np.conj(np.transpose(ucx)),np.dot(H_clock.site_ops[1],ucx))

# #z in new basis
# z_spin_basis = np.zeros(pxp.dim,dtype=complex)
# goes_to = dict()
# for n in range(0,pxp.base):
    # state = np.zeros(pxp.base)
    # state[n] = 1
    # goes_to[n] = np.dot(np.conj(np.transpose(ucx)),state)
    
# #10101 state
# z_spin_basis = np.zeros(pxp.dim,dtype=complex)
# for n in range(0,np.size(pxp.basis,axis=0)):
    # state_bits = np.copy(pxp.basis[n])
    # coef = 1
    # for m in range(0,np.size(state_bits,axis=0)):
        # if m % 2 == 0:
            # coef = coef * goes_to[0][state_bits[m]]
        # else:
            # coef = coef * goes_to[1][state_bits[m]]
    # z_spin_basis[n] = coef
# z_spin_basis = z_spin_basis / np.power(np.vdot(z_spin_basis,z_spin_basis),0.5)

# H = Hamiltonian(pxp,pxp_syms)
# H.site_ops[1] = X
# H.site_ops[2] = P0_spin_basis
# H.model = np.array([[2,1,2]])
# H.model_coef = np.array([1])
# H.gen()
# H.sector.find_eig()
# print(H.sector.eigvalues())
# print(H_clock.sector.eigvalues())

# #fsa basis
# c=np.real(X[0,1])
# Sp = np.array([[0,c,0],[0,0,c],[0,0,0]]).transpose()
# Sm = np.conj(np.transpose(Sp))

# #P+P on even sites
# pe = Hamiltonian(pxp,pxp_syms)
# pe.site_ops[1] = Sp
# pe.site_ops[2] = P0_spin_basis
# pe.model = np.array([[2,1,2]])
# pe.model_coef = np.array([1])
# pe.gen(parity=1)

# #P-P on odd sites
# mo = Hamiltonian(pxp,pxp_syms)
# mo.site_ops[1] = Sm
# mo.site_ops[2] = P0_spin_basis
# mo.model = np.array([[2,1,2]])
# mo.model_coef = np.array([1])
# mo.gen(parity=0)

# Hp = H_operations.add(pe,mo,np.array([1,1]))
# Hp = Hp.sector.matrix()
# Hm = np.conj(np.transpose(Hp))

# fsa_basis = z_spin_basis
# current_state = fsa_basis
# fsa_dim = 2*pxp.N
# for n in range(0,fsa_dim):
    # next_state = np.dot(Hp,current_state)
    # next_state = next_state/np.power(np.vdot(next_state,next_state),0.5)
    # fsa_basis = np.vstack((fsa_basis,next_state))
    # current_state = next_state
# fsa_basis = np.transpose(fsa_basis)
# fsa_basis,temp = np.linalg.qr(fsa_basis)
# # gs = gram_schmidt(fsa_basis)
# # gs.ortho()
# # fsa_basis = gs.ortho_basis
    
# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# plt.matshow(np.abs(H_fsa))
# plt.show()
# e,u = np.linalg.eigh(H_fsa)
# fsa_energy = np.conj(u[0,:])
# overlap_fsa = np.log10(np.abs(fsa_energy)**2)
# plt.scatter(e,overlap_fsa,marker="x",s=100,color="red",label="Rotated FSA")
    
# z_spin_energy_basis = np.dot(np.conj(np.transpose(H.sector.eigvectors())),z_spin_basis)
# overlap = np.log10(np.abs(z_spin_energy_basis)**2)
# plt.scatter(H.sector.eigvalues(),overlap)
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# plt.title(r"$N_c=3$ Clock Rotated, $H=P^{'}XP^{'}, N=$"+str(pxp.N))
# plt.legend()
# plt.show()

# # t=np.arange(0,20,0.01)
# # f=np.zeros(np.size(t))
# # for n in range(0,np.size(t,axis=0)):
    # # evolved_state = time_evolve_state(z_spin_energy_basis,H.sector.eigvalues(),t[n])
    # # f[n] = np.abs(np.vdot(evolved_state,z_spin_energy_basis))**2
# # fidelity(z,H_clock).plot(t,z)
# # plt.plot(t,f,alpha=0.8,linestyle="--")
# # plt.show()

# print("\n")
# u_fsa_comp = np.dot(fsa_basis,u)
# for n in range(0,np.size(u_fsa_comp,axis=1)):
    # # u_fsa_comp[:,n] = u_fsa_comp[:,n] / np.power(np.vdot(u_fsa_comp[:,n],u_fsa_comp[:,n]),0.5)
    # print(np.vdot(u_fsa_comp[:,n],u_fsa_comp[:,n]))
    
# exact_overlap = np.zeros(np.size(e))
# for n in range(0,np.size(e,axis=0)):
    # max_overlap = 0
    # for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        # temp = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))
        # if temp > max_overlap:
            # max_overlap = temp
    # exact_overlap[n] = max_overlap
# print(exact_overlap)
# plt.plot(e,exact_overlap,marker="s")
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
# plt.title(r"$N_c=3$ Clock Rotated, $H=P^{'}XP^{'}$, Rotated FSA, $N=$"+str(pxp.N))
# plt.show()
        
