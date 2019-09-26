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

def find_hamming_sectors(state_bits):
    #organize states via hamming distance from Neel
    hamming_sectors = dict()
    for n in range(0,pxp.N+1):
        hamming_sectors[n] = []
    for n in range(0,pxp.dim):
        h = 0
        for m in range(0,pxp.N,1):
            if pxp.basis[n][m] != state_bits[m]:
                h = h+1
        hamming_sectors[int(h)] = np.append(hamming_sectors[int(h)],pxp.basis_refs[n])
    return hamming_sectors

#init small hypercube
N=12
pxp = unlocking_System([0,1],"periodic",2,N)
pxp.gen_basis()

H_total = np.load("./disorder_realizations/H,001.npy")
c=1
pbar=ProgressBar()
for n in pbar(range(2,10)):
    temp = np.load("./disorder_realizations/H,00"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1
pbar=ProgressBar()
for n in pbar(range(10,100)):
    temp = np.load("./disorder_realizations/H,0"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1

pbar=ProgressBar()
for n in pbar(range(100,501)):
    temp = np.load("./disorder_realizations/H,"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1
print(c)
H_total = H_total / c
print(H_total)
np.save("H_avg,cycle_disorder,12",H_total)

e,u = np.linalg.eigh(H_total)
z=zm_state(2,1,pxp)
z_energy = np.conj(u[pxp.keys[z.ref],:])
t=np.arange(0,10,0.1)
f=np.zeros(np.size(t))
for n in range(0,np.size(f,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.show()
pbar=ProgressBar()

for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
# for n in pbar(range(0,np.size(e,axis=0))):
    z_energy = np.conj(u[n,:])
    f=np.zeros(np.size(t))
    for m in range(0,np.size(f,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[m])
        f[m] = np.abs(np.vdot(evolved_state,z_energy))**2

    for m in range(0,np.size(f,axis=0)):
        if f[m] <0.2:
            cut = m
            break
    f_max = np.max(f[cut:])
    plt.plot(t,f)
plt.title(r"$\textrm{Hypercube, Cycle Disorder, Quench Fidelity}, N=12$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2 $")
plt.show()

z=zm_state(2,1,pxp)
z_energy = np.conj(u[pxp.keys[z.ref],:])
overlap = np.log10(np.abs(z_energy)**2)
eigenvalues = np.copy(e)
to_del=[]
for m in range(0,np.size(overlap,axis=0)):
    if overlap[m] <-5:
        to_del = np.append(to_del,m)
for m in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[m])
    eigenvalues=np.delete(eigenvalues,to_del[m])
plt.scatter(eigenvalues,overlap)
plt.title(r"$\textrm{Hypercube, Cycle Disorder, Eigenstate overlap with Neel}, N=12$")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log \vert \langle E \vert \psi \rangle \vert^2$")
plt.show()
plt.show()

r_diff = np.zeros(np.size(e)-1)
for n in range(0,np.size(r_diff,axis=0)):
    r_diff[n] = np.abs(e[n+1]-e[n])

level_ratios = np.zeros(np.size(r_diff)-1)
for n in range(0,np.size(level_ratios,axis=0)):
    level_ratios[n] = np.min(np.array((r_diff[n],r_diff[n+1])))/np.max(np.array((r_diff[n],r_diff[n+1])))
mean_level_ratios = np.sum(level_ratios)/np.size(level_ratios)
print(mean_level_ratios)
