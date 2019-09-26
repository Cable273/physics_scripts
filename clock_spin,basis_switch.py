#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian
from System_Classes import unlocking_System
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state
from rw_functions import save_obj,load_obj

pxp = unlocking_System([0],"periodic",3,8)
pxp.gen_basis()
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])

H1=clock_Hamiltonian(pxp,pxp_syms)
H2=spin_Hamiltonian(pxp,"x",pxp_syms)
e_clock,u_clock = np.linalg.eigh(H1.site_ops[1])
e_spin,u_spin = np.linalg.eigh(H2.site_ops[1])

spin2clock_u = np.dot(u_spin,np.conj(np.transpose(u_clock)))
clock2spin_u = np.dot(u_clock,np.conj(np.transpose(u_spin)))
root6 = np.power(6,0.5)
root23 = np.power(2/3,0.5)
clock2spin_u = np.array([[1/root6,root23,-1/root6],[1/root6-1j/2,-1/root6,-1/root6-1j/2],[1/root6+1j/2,-1/root6,-1/root6+1j/2]])
print(np.shape(spin2clock_u))
print(np.shape(clock2spin_u))
P = np.array([[1,0,0],[0,0,0],[0,0,0]])
# P = np.array([[1,0,0],[0,0,0],[0,0,0]])
P_clock_spin_basis = np.dot(np.conj(np.transpose(clock2spin_u)),np.dot(P,clock2spin_u))
P_spin_clock_basis = np.dot(np.conj(np.transpose(spin2clock_u)),np.dot(P,spin2clock_u))
# print(clock2spin_u)
# print(P_clock_spin_basis)
print(P_clock_spin_basis)

def com(a,b):
    return np.dot(a,b)-np.dot(b,a)

sp = np.array([[0,0,0],[1,0,0],[0,1,0]])
sm = np.array([[0,1,0],[0,0,1],[0,0,0]])
print(P)
print("\n")
print(sp)
print(com(sp,P))
print("\n")
print(sm)
print(com(sm,P))

e,u = np.linalg.eigh(P_spin_clock_basis)

z=zm_state(2,1,pxp)
Neel_state_spin_basis = np.zeros(np.size(pxp.basis_refs))
Neel_state_spin_basis[pxp.keys[z.ref]] = 1
Neel_state_clock_basis = np.dot(spin2clock_u,Neel_state_spin_basis)

z=zm_state(2,1,pxp)
Neel_state_clock_basis = np.zeros(np.size(pxp.basis_refs))
Neel_state_clock_basis[pxp.keys[z.ref]] = 1
Neel_state_spin_basis = np.dot(clock2spin_u,Neel_state_clock_basis)
print(Neel_state_spin_basis)


H = Hamiltonian(pxp,pxp_syms)
H.site_ops[2] = P_spin_clock_basis
H.site_ops[1] = H1.site_ops[1]
H.model=np.array(([[2,1,2]]))
H.model_coef=np.array((1))
H.gen()
H.sector.find_eig()
print(H.sector.eigvalues())
# eig_overlap.plot(z.bits,H)
# plt.show()
# fidelity.plot(z.bits,np.arange(0,20,0.01),H)
# plt.show()

H_clock = Hamiltonian(pxp,pxp_syms)
H_clock.site_ops[2] = P
H_clock.site_ops[1] = H2.site_ops[1]
H_clock.model = np.array([[2,1,2]])
H_clock.model_coef = np.array((1))

# H_clock = clock_Hamiltonian(pxp,pxp_syms)
H_clock.gen()
H_clock.sector.find_eig()
print(H_clock.sector.eigvalues())

# eig_overlap.plot(z.bits,H_clock)
# plt.show()
# fidelity.plot(z.bits,np.arange(0,20,0.01),H_clock)
# plt.show()


