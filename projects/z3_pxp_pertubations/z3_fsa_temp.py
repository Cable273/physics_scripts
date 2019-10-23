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
from System_Classes import unlocking_System,U1_system
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import *

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

Hp = np.zeros((pxp.dim,pxp.dim))
Hp2 = np.zeros((pxp.dim,pxp.dim))
pbar=ProgressBar()
for n in pbar(range(0,pxp.dim)):
    bits = pxp.basis[n]
    for m in range(0,pxp.N):
        if m == pxp.N-1:
            mp1=0
            mp2=1
        elif m == pxp.N-2:
            mp1=m+1
            mp2=0
        else:
            mp1=m+1
            mp2=m+2

        if m == 0:
            mm1 = pxp.N-1
            mm2 = pxp.N-2
        elif m == 1:
            mm1 = m-1
            mm2 = pxp.N-1
        else:
            mm1 = m-1
            mm2 = m-2

        if m % 3 == 0:
            if bits[m] == 0:
                new_bits = np.copy(bits)
                new_bits[m] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    Hp[pxp.keys[new_ref],n] = Hp[pxp.keys[new_ref],n] +1
                    # Hp2[pxp.keys[new_ref],n] = Hp2[pxp.keys[new_ref],n] + 1

        if m % 3 == 1:
            if bits[m] == 1:
                new_bits = np.copy(bits)
                new_bits[m] = 0
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    Hp[pxp.keys[new_ref],n] = Hp[pxp.keys[new_ref],n] +1
                    # Hp2[pxp.keys[new_ref],n] = Hp2[pxp.keys[new_ref],n] + 1

        if m % 3 == 2:
            if bits[m] == 1:
                new_bits = np.copy(bits)
                new_bits[m] = 0
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    Hp[pxp.keys[new_ref],n] = Hp[pxp.keys[new_ref],n] + 1

Hm = np.conj(np.transpose(Hp))
def com(a,b):
    return np.dot(a,b)-np.dot(b,a)
Hz = 1/2*com(Hp,Hm)
# plt.matshow(np.abs(Hz))
# plt.show()
# print((np.abs(Hz-np.conj(np.transpose(Hz)))<1e-5).all())
H = spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()

z=zm_state(3,1,pxp,2)
k=pxp_syms.find_k_ref(z.ref)
U = dict()
for n in range(0,np.size(k,axis=0)):
    U[n] = pxp_syms.basis_transformation(k[n])

fsa_basis = z.prod_basis()
fsa_basis2 = z.prod_basis()
krylov_basis = z.prod_basis()
current_state = fsa_basis
current_state2 = fsa_basis
current_stateK = fsa_basis
fsa_dim = int(2*N/3)
for n in range(0,fsa_dim):
    next_state = np.dot(Hp,current_state)
    next_state2 = np.dot(Hp2,current_state2)
    next_stateK = np.dot(H.sector.matrix(),current_stateK)

    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    next_state2 = next_state2 / np.power(np.vdot(next_state2,next_state2),0.5)
    next_stateK = next_stateK / np.power(np.vdot(next_stateK,next_stateK),0.5)

    fsa_basis = np.vstack((fsa_basis,next_state))
    fsa_basis2 = np.vstack((fsa_basis2,next_state2))
    krylov_basis = np.vstack((krylov_basis,next_stateK))

    current_state = next_state
    current_state2 = next_state2
    current_stateK = next_stateK

fsa_basis = np.transpose(fsa_basis)
psi = fsa_basis[:,np.size(fsa_basis,axis=1)-1]
from Diagnostics import print_wf
for n in range(0,np.size(fsa_basis,axis=0)):
    print("\n")
    print_wf(fsa_basis[:,n],pxp,1e-5)
# krylov_basis = np.transpose(krylov_basis)
# gs = gram_schmidt(krylov_basis)
# gs.ortho()
# krylov_basis = gs.ortho_basis
# # krylov_basis,temp = np.linalg.qr(krylov_basis)
# # fsa_basis2 = np.transpose(fsa_basis2)
# # fsa_basis = fsa_basis2
# # fsa_basis = np.hstack((fsa_basis,fsa_basis2))
# # from Calculations import gram_schmidt
# # gs = gram_schmidt(fsa_basis)
# # gs.ortho()
# # fsa_basis = gs.ortho_basis


# for n in range(0,np.size(k,axis=0)):
    # H.gen(k[n])
    # H.sector.find_eig(k[n])
    # eig_overlap(z,H,k[n]).plot()

# H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
# H_krylov = np.dot(np.conj(np.transpose(krylov_basis)),np.dot(H.sector.matrix(),krylov_basis))
# e,u = np.linalg.eigh(H_fsa)
# ek,uk = np.linalg.eigh(H_krylov)
# plt.scatter(e,np.log10(np.abs(u[0,:])**2),marker="D",s=100,label="FSA",alpha=0.6)
# plt.scatter(ek,np.log10(np.abs(uk[0,:])**2),marker="x",color="red",s=100,label="Krylov")
# plt.legend()
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\log(\vert \langle \psi \vert E \rangle \vert^2)$")
# plt.title(r"$Z_3$ Scar approximations, N="+str(pxp.N))
# plt.show()

# t=np.arange(0,20,0.01)
# f_fsa = np.zeros(np.size(t))
# psi_fsa_energy = np.conj(u[0,:])
# for n in range(0,np.size(t,axis=0)):
    # evolved_state = time_evolve_state(psi_fsa_energy,e,t[n])
    # f_fsa[n] = np.abs(np.vdot(evolved_state,psi_fsa_energy))**2
# fidelity(z,H,"use sym").plot(t,z)
# plt.plot(t,f_fsa,label="FSA Projected Dynamics")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
# plt.show()

# u_exact_comp = np.dot(U[0],H.sector.eigvectors(k[0]))
# u_exact_comp = np.hstack((u_exact_comp,np.dot(U[1],H.sector.eigvectors(k[1]))))
# u_exact_comp = np.hstack((u_exact_comp,np.dot(U[2],H.sector.eigvectors(k[2]))))

# u_comp = np.dot(fsa_basis,u)
# uk_comp = np.dot(krylov_basis,uk)
# exact_overlap = np.zeros(np.size(e))

# for n in range(0,np.size(u_comp,axis=1)):
    # max_overlap = 0
    # for m in range(0,np.size(u_exact_comp,axis=1)):
        # temp = np.abs(np.vdot(u_comp[:,n],u_exact_comp[:,m]))**2
        # if temp > max_overlap:
            # max_overlap = temp
    # exact_overlap[n] = max_overlap

# exact_overlapk = np.zeros(np.size(ek))
# for n in range(0,np.size(uk_comp,axis=1)):
    # max_overlap = 0
    # for m in range(0,np.size(u_exact_comp,axis=1)):
        # temp = np.abs(np.vdot(uk_comp[:,n],u_exact_comp[:,m]))**2
        # if temp > max_overlap:
            # max_overlap = temp
    # exact_overlapk[n] = max_overlap

# plt.plot(e,exact_overlap,marker="s",label="FSA")
# plt.plot(ek,exact_overlapk,marker="s",label="Krylov")
# plt.xlabel(r"$E$")
# plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi_{exact} \rangle \vert^2$")
# plt.title(r"$Z_3$ Scar approximations, N="+str(pxp.N))
# plt.legend()
# plt.show()
