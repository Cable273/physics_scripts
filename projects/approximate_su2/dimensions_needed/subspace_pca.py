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

N_vals=np.arange(12,14,2)
delta_t = 0.1
t_max_vals = np.arange(0,10,0.1)
error_tol = 0.01
n_necessary = np.zeros((np.size(N_vals),np.size(t_max_vals)))
# n_necessary = np.zeros(np.size(t_max_vals))
# n_necessary = 0
for count in range(0,np.size(N_vals,axis=0)):
    N = N_vals[count]
    N_system = int(N_vals[0])
    pxp = unlocking_System([0],"periodic",2,N)
    pxp.gen_basis()
    pxp_syms = model_sym_data(pxp,[translational(pxp)])

    H = spin_Hamiltonian(pxp,"x")
    H.gen()
    H.sector.find_eig()
    z=zm_state(2,1,pxp)
    z_energy = H.sector.eigvectors()[pxp.keys[z.ref],:]

    pbar=ProgressBar()
    for index in pbar(range(0,np.size(t_max_vals,axis=0))):
        t_max = t_max_vals[index]
        t=np.arange(0,t_max+delta_t,delta_t)
        L = t_max/delta_t
        M = np.zeros((pxp.dim,np.size(t)),dtype=complex)
        psi_time_step = dict()
        for n in range(0,np.size(t,axis=0)):
            evolved_state = time_evolve_state(z_energy,H.sector.eigvalues(),t[n])
            M[:,n]  = evolved_state
        # plt.matshow(np.abs(M))
        # plt.show()

        u,s,vh = np.linalg.svd(M)
        def error(N):
            return np.sum(s[N:])
        N = np.arange(0,np.size(s)).astype(int)
        e = np.zeros(np.size(N))
        for n in range(0,np.size(e,axis=0)):
            e[n] = error(N[n])

        #find n such that error<0.01*L
        for n in range(0,np.size(e,axis=0)):
            if e[n]<error_tol*L:
                n_necessary[count,index] = n
                # n_necessary = n
                break
        # if n_necessary == int(N_vals[count]+1):
            # break

for n in range(0,np.size(N_vals,axis=0)):
    plt.plot(t_max_vals,n_necessary[n,:],label=r"$N=$"+str(N_vals[n]))
plt.legend()
plt.xlabel(r"$t_{max}$")
plt.ylabel(r"$\tilde{N}$")
plt.title(r"$\sum_{k>\tilde{N}} \vert \sigma_k \vert^2=$"+str(error_tol)+"$\, L, \Delta t =$"+str(delta_t))
plt.show()


# plt.plot(t_max_vals,n_necessary)
# plt.show()
svd_vectors = u[:,:int(N_system+1)]
H_svd = np.dot(np.conj(np.transpose(svd_vectors)),np.dot(H.sector.matrix(),svd_vectors))
# plt.matshow(np.abs(H_svd))
# plt.show()
e,u = np.linalg.eigh(H_svd)

z=zm_state(2,1,pxp)
eig_overlap(z,H).plot()
# plt.show()

z_svd = np.dot(np.conj(np.transpose(svd_vectors)),z.prod_basis())
z_svd = z_svd / np.power(np.vdot(z_svd,z_svd),0.5)
z_svd_energy = np.dot(np.conj(np.transpose(u)),z_svd)
overlap = np.log10(np.abs(z_svd_energy)**2)
plt.scatter(e,overlap,marker="s",s=100,color="red")
plt.show()

t=np.arange(0,5,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(overlap,axis=0)):
    evolved_state = time_evolve_state(z_svd_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_svd_energy))**2
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$\textrm{PXP, N=16, D=17 SVD Basis}$")
plt.show()
