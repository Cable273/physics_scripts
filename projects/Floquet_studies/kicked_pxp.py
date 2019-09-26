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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection

pxp = unlocking_System([0,1],"periodic",2,10)
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])

#pauli ops
X = Hamiltonian(pxp,pxp_syms)
Y = Hamiltonian(pxp,pxp_syms)
Z = Hamiltonian(pxp,pxp_syms)

X.site_ops[1] = np.array([[0,1],[1,0]])
Y.site_ops[1] = np.array([[0,-1j],[1j,0]])
Z.site_ops[1] = np.array([[-1,0],[0,1]])

X.model,X.model_coef = np.array([[1]]),np.array((1))
Y.model,Y.model_coef = np.array([[1]]),np.array((1))
Z.model,Z.model_coef = np.array([[1]]),np.array((1))

X.gen()
Y.gen()
Z.gen()

#n_i n_i+1
H0 = Hamiltonian(pxp,pxp_syms)
H0.site_ops[1] = np.array([[0,0],[0,1]])
H0.model = np.array([[1,1]])
H0.model_coef = np.array((1))

#X_n
H_kick = Hamiltonian(pxp,pxp_syms)
H_kick.site_ops[1] = np.array([[0,1],[1,0]])
H_kick.model = np.array([[1]])
H_kick.model_coef = np.array((1))

H0.gen()
H_kick.gen()
H0.sector.find_eig()
H_kick.sector.find_eig()

tau = 2*math.pi/20
V = 0.05

U0 = np.dot(H0.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*tau*H0.sector.eigvalues())),np.conj(np.transpose(H0.sector.eigvectors()))))
U_kick = np.dot(H_kick.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*V*H_kick.sector.eigvalues())),np.conj(np.transpose(H_kick.sector.eigvectors()))))

F = np.dot(U_kick,U0)
e,u = np.linalg.eig(F)

z=zm_state(2,1,pxp)
z1=ref_state(0,pxp)
z2=ref_state(1,pxp)
no_steps = 500
f = np.zeros(no_steps)
f1 = np.zeros(no_steps)
f2 = np.zeros(no_steps)
current_state = z.prod_basis()
current_state1 = z1.prod_basis()
current_state2 = z2.prod_basis()
for n in range(0,no_steps):
    f[n] = np.abs(np.vdot(current_state,z.prod_basis()))**2
    f1[n] = np.abs(np.vdot(current_state1,z1.prod_basis()))**2
    f2[n] = np.abs(np.vdot(current_state2,z2.prod_basis()))**2
    current_state = np.dot(F,current_state)
    current_state1 = np.dot(F,current_state1)
    current_state2 = np.dot(F,current_state2)

t=np.arange(0,no_steps*tau,tau)
plt.plot(t,f,label=r"$\vert 0101... \rangle $")
plt.plot(t,f1,label=r"$ \vert 0000... \rangle $")
plt.plot(t,f2,label=r"$ \vert 1000... \rangle $")

plt.xlabel(r"$t$")
plt.ylabel(r"$F$")
plt.legend()
plt.title(r"PXP Floquet, Fidelity, N="+str(pxp.N))
plt.show()
