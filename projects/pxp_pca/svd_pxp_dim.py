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

N = 12
pxp = unlocking_System([0],"periodic",2,N)
pxp.gen_basis()
pxp_syms=model_sym_data(pxp,[translational(pxp)])

H=spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()

delta_t = 0.1
tol = 0.01
t_max_vals = np.arange(0.01,4.01,0.01)
N_kept = np.zeros(np.size(t_max_vals))
# N_kept = None
pbar=ProgressBar()
for t_index in pbar(range(1,np.size(t_max_vals,axis=0))):
    t_max = t_max_vals[t_index]
    L = t_max/delta_t

    t=np.arange(0,t_max+delta_t,delta_t)
    M = np.zeros((pxp.dim,np.size(t)),dtype=complex)

    z=zm_state(2,1,pxp)
    z_energy = np.conj(H.sector.eigvectors()[pxp.keys[z.ref],:])
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(z_energy,H.sector.eigvalues(),t[n])
        evolved_state_comp = np.dot(H.sector.eigvectors(),evolved_state)
        M[:,n] = evolved_state_comp

    U,S,Vh = np.linalg.svd(M)

    #for plotting gif (vertical line)
    # s_diff = np.zeros(np.size(S)-1)
    # for n in range(0,np.size(S,axis=0)-1):
        # s_diff[n] = np.log10(S[n+1])-np.log10(S[n])
    # min_arg = np.argmin(s_diff)+1
    # plt.plot(np.log10(S),label=str(min_arg))
    # plt.xlabel(r"$n$")
    # plt.ylabel(r"$\log(\sigma_n)$")
    # # plt.title(r"Singular values, $t_{max}=$"+str("{0:.2f}".format(t_max))+", $\Delta t=$"+str(delta_t)+", $N=$"+str(N))
    # plt.title(r"$t_{max}=$"+str("{0:.2f}".format(t_max))+", $\Delta t=$"+str(delta_t)+", $N=$"+str(N))
    # plt.axvline(x=min_arg)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("./gif/img"+str(t_index))
    # plt.cla()
    # plt.show()

    def cut_error(N_k):
        return np.sum(np.abs(S[N_k:])**2)

    if t_index > 0:
        for n in range(int(N_kept[t_index-1]),np.size(S)):
            error = cut_error(n)
            if error <= tol*L:
                # N_kept = n
                N_kept[t_index] = n
                break
    else:
        for n in range(0,np.size(S)):
            error = cut_error(n)
            if error <= tol*L:
                # N_kept = n
                N_kept[t_index] = n
                break
    # if N_kept is not None:
        # basis = U[:,:N_kept+1]
        # np.save("./svd_basis_for_gif/svd_basis,16,"+str(t_index),basis)
        # t_info = np.array((t_max,N_kept))
        # np.save("./svd_basis_for_gif/t_info,16,"+str(t_index),t_info)

    # print(basis)
plt.plot(t_max_vals,N_kept)
plt.xlabel(r"$t_{max}$")
plt.ylabel(r"$N_{kept}$")
plt.title(r"SVD vectors to keep such that error $\sum_{k>N_k} \vert \sigma \vert_k^2 =$"+str(tol)+"L, PXP, N="+str(pxp.N))
plt.show()

# take first N+1 vecs, project ham to basis, get scars?
# basis = U[:,:N+1+(N+1-4)]
basis = U[:,:N+1]
# basis = U[:,:7]
H=spin_Hamiltonian(pxp,"x",pxp_syms)
H.gen()
H.sector.find_eig()
z=zm_state(2,1,pxp)
psi = z.prod_basis()

H_rot = np.dot(np.conj(np.transpose(basis)),np.dot(H.sector.matrix(),basis))
psi_rot = np.dot(np.conj(np.transpose(basis)),psi)
e,u = np.linalg.eigh(H_rot)

overlap = np.zeros(np.size(e))
for n in range(0,np.size(e,axis=0)):
    overlap[n] = np.log10(np.abs(np.vdot(psi_rot,u[:,n]))**2)
eig_overlap(z,H).plot()
plt.scatter(e,overlap,marker="x",color="red",s=100,label="SVD")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle \psi_{svd} \vert \psi_{neel, rot} \rangle \vert^2)$")
print(t_max)
plt.title(r"SVD Projection eigenstates $t_{max}=4$, tol=$0.01L$, overlap with Neel, PXP, N="+str(pxp.N))
plt.show()

u_comp = np.dot(basis,u)
exact_overlap = np.zeros(np.size(e))
for n in range(0,np.size(u_comp,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp = np.abs(np.vdot(u_comp[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    exact_overlap[n] = max_overlap
plt.scatter(e,exact_overlap,label="SVD eig overlap with Exact eig")

Hp = np.zeros((pxp.dim,pxp.dim))
for n in range(0,np.size(pxp.basis,axis=0)):
    state_bits = np.copy(pxp.basis[n])
    for m in range(0,np.size(state_bits,axis=0)):
        if m % 2 == 0:
            if state_bits[m] == 1:
                new_bits = np.copy(state_bits)
                new_bits[m] = 0
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    new_index = pxp.keys[ new_ref ]
                    Hp[n,new_index] = 1
        else:
            if state_bits[m] == 0:
                new_bits = np.copy(state_bits)
                new_bits[m] = 1
                new_ref = bin_to_int_base_m(new_bits,pxp.base)
                if new_ref in pxp.basis_refs:
                    new_index = pxp.keys[ new_ref ]
                    Hp[n,new_index] = 1
Hm = np.conj(np.transpose(Hp))
fsa_dim = int(pxp.N)
z=zm_state(2,1,pxp)
fsa_basis = z.prod_basis()
current_state = fsa_basis
for n in range(0,fsa_dim):
    next_state = np.dot(Hm,current_state)
    next_state = next_state / np.power(np.vdot(next_state,next_state),0.5)
    fsa_basis = np.vstack((fsa_basis,next_state))
    current_state = next_state
fsa_basis = np.transpose(fsa_basis)

H_fsa = np.dot(np.conj(np.transpose(fsa_basis)),np.dot(H.sector.matrix(),fsa_basis))
e_fsa,u_fsa = np.linalg.eigh(H_fsa)
u_fsa_comp = np.dot(fsa_basis,u_fsa)

svd_fsa_overlap = np.zeros(np.size(e))
for n in range(0,np.size(u_comp,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(u_fsa_comp,axis=1)):
        temp = np.abs(np.vdot(u_fsa_comp[:,m],u_comp[:,n]))**2
        if temp > max_overlap:
            max_overlap = temp
    svd_fsa_overlap[n] = max_overlap

fsa_exact_overlap = np.zeros(np.size(e_fsa))
for n in range(0,np.size(u_fsa_comp,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H.sector.eigvectors(),axis=1)):
        temp = np.abs(np.vdot(u_fsa_comp[:,n],H.sector.eigvectors()[:,m]))**2
        if temp > max_overlap:
            max_overlap = temp
    fsa_exact_overlap[n] = max_overlap

plt.scatter(e,svd_fsa_overlap,marker="x",label="SVD eig Overlap with FSA eig")
plt.scatter(e_fsa,fsa_exact_overlap,marker="s",alpha=0.6,label="FSA eig Overlap with Exact eig")
plt.xlabel(r"$E$")
plt.title(r"SVD Projection eigenstates, PXP, N="+str(pxp.N))
plt.ylabel(r"$\vert \langle \psi_{approx} \vert \psi \rangle \vert^2$")
plt.legend()
plt.show()
