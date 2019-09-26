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

N=6
pxp = unlocking_System([0,1],"periodic",2,N)
pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
pxp_half = unlocking_System([0],"periodic",2,N)

#form Rydberg subspace in full basis
rydberg_subspace = np.zeros((pxp.dim,pxp_half.dim))
for n in range(0,pxp_half.dim):
    rydberg_subspace[pxp.keys[pxp_half.basis_refs[n]],n]=1

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

# for tau in range
c=0
pbar=ProgressBar()
tau_vals = np.arange(0.01,5,0.001)
theta_vals=np.zeros(np.size(tau_vals))
for tau in pbar(np.arange(0.01,5,0.001)):
    a=2*math.pi/tau
    # tau = 2*math.pi/(4/3)
    # print(tau)
    # tau = 4.71
    # tau = 2*math.pi/2
    # tau = 0.1
    V = 0.05

    U0 = np.dot(H0.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*tau*H0.sector.eigvalues())),np.conj(np.transpose(H0.sector.eigvectors()))))
    U_kick = np.dot(H_kick.sector.eigvectors(),np.dot(np.diag(np.exp(-1j*V*H_kick.sector.eigvalues())),np.conj(np.transpose(H_kick.sector.eigvectors()))))

    F = np.dot(U_kick,U0)
    e,u = np.linalg.eig(F)
    u,temp = np.linalg.qr(u)
    #form subsapce of Floquet eigenvectors, same dimension as Rydberg basis with largest weight in that basis
    subspace_eig_norm = np.zeros(np.size(u,axis=1))
    for n in range(0,np.size(u,axis=0)):
        projected_state = np.dot(np.transpose(rydberg_subspace),u[:,n])
        subspace_eig_norm[n] = np.real(np.vdot(projected_state,projected_state))
    # plt.plot(np.sort(subspace_eig_norm))
    # plt.title(r"$V=$"+str(V)+r"$, \tau=$"+str(tau))
    # plt.ylabel(r"$\vert P^{\dagger} \vert \psi \rangle \vert^2$")
    # plt.savefig("./gif/floquet_weights,"+str(c)+".png")
    # c = c+1
    # plt.cla()

    eig_indices = np.arange(0,np.size(u,axis=1))
    subspace_eig_norm, eig_indices = (list(t) for t in zip(*sorted(zip(subspace_eig_norm, eig_indices))))
    subspace_eig_norm = np.flip(subspace_eig_norm)
    eig_indices = np.flip(eig_indices)

    floquet_subspace = np.zeros((pxp.dim,pxp_half.dim),dtype=complex)
    for n in np.arange(0,pxp_half.dim):
        floquet_subspace[:,n] = u[:,eig_indices[n]]

    from Diagnostics import check_orthornormal,is_hermitian,is_unitary

    #form orthonromal basis using qr decomp
    # floquet_subspace,temp = np.linalg.qr(u)
    S = np.dot(np.conj(np.transpose(floquet_subspace)),rydberg_subspace)
    U,s,vh = np.linalg.svd(S)
    cos_products = 1
    for n in range(0,np.size(s,axis=0)):
        cos_products = cos_products * s[n]
    theta = np.arccos(cos_products)
    theta_vals[c] = theta
    c=c+1
plt.title("Kicked PXP, Angle Between Subspaces. $V=$"+str(V)+", N="+str(N))
plt.ylabel(r"$\theta_{A,B}$")
plt.xlabel(r"$\tau$")
plt.plot(tau_vals,theta_vals)
plt.show()
