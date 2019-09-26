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
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern'],'size':26})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# matplotlib.rcParams['figure.dpi'] = 400

N=6
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()


def com(A,B):
    return np.dot(A,B)-np.dot(B,A)

H = spin_Hamiltonian(pxp,"x")
H.gen()

P = np.zeros((pxp.dim,pxp.dim))
pxp_constrained = unlocking_System([0],"periodic",2,N)
pxp_constrained.gen_basis()
for n in range(0,np.size(pxp_constrained.basis_refs,axis=0)):
    index = pxp.keys[pxp_constrained.basis_refs[n]]
    P[index,index] = 1

linblads=dict()
for j in range(0,pxp.N):
    linblads[j] = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        if pxp.basis[n][j] == 0:
            linblads[j][n,n] = 1

linblads = dict()
for j in range(0,pxp.N):
    if j == pxp.N-1:
        jp1=0
    else:
        jp1 = j + 1

    temp = np.zeros((pxp.dim,pxp.dim))
    for n in range(0,np.size(pxp.basis_refs,axis=0)):
        if pxp.basis[n][j] == 1 and pxp.basis[n][jp1] == 1:
            temp[n,n] = 1

    linblads[j] = np.eye(pxp.dim)-temp
    # linblads[j] = temp



def linblad(rho,H,P,coef):
    temp = -1j * com(H,rho)
    for n in range(0,len(linblads)):
        temp = temp + coef*(np.dot(linblads[n],np.dot(rho,linblads[n]))-0.5*np.dot(linblads[n],rho)-0.5*np.dot(rho,linblads[n])) 
    return temp
    # return -1j * com(H,rho)+coef*(np.dot(P,np.dot(rho,P))-0.5*np.dot(P,rho)-0.5*np.dot(rho,P)) 

pbar=ProgressBar()
for index in pbar(range(0,np.size(pxp.basis_refs),2)):
    z=ref_state(pxp.basis_refs[index],pxp)
    index = pxp.keys[z.ref]
    rho = np.zeros((pxp.dim,pxp.dim))
    rho[index,index] = 1

    #integrate linblad with runge kutta
    rho
    delta_t = 0.1
    t_max=10
    t=np.arange(0,t_max+delta_t,delta_t)
    rho_t=dict()
    rho_t[0] = rho
    coef=0.1
    f = np.zeros(np.size(t))
    f[0] = 1
    z_anti = zm_state(2,1,pxp,1)
    rho_anti = np.zeros((pxp.dim,pxp.dim))
    index=pxp.keys[z_anti.ref]
    rho_anti[index,index] = 1

    for n in range(1,np.size(t,axis=0)):
        k1 = delta_t * linblad(rho_t[n-1],H.sector.matrix(),P,coef)
        k2 = delta_t * (linblad(rho_t[n-1]+k1/2,H.sector.matrix(),P,coef))
        k3 = delta_t * (linblad(rho_t[n-1]+k2/2,H.sector.matrix(),P,coef))
        k4 = delta_t * (linblad(rho_t[n-1]+k3,H.sector.matrix(),P,coef))
        rho_t[n] = rho_t[n-1] + 1/6 *(k1+2*k2+2*k3+k4)
        f[n] = np.real(np.trace(np.dot(np.conj(np.transpose(rho_t[n])),rho_t[0])))
    plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"$\textrm{Hypercube, Linblad (Measuring neighbouring excitations)}, N=6$")
plt.show()
