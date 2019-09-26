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

H_total = np.load("./disorder_realizations/bond_disorder/H,001.npy")
c=1
pbar=ProgressBar()
for n in pbar(range(2,10)):
    temp = np.load("./disorder_realizations/bond_disorder/H,00"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1
pbar=ProgressBar()
for n in pbar(range(10,100)):
    temp = np.load("./disorder_realizations/bond_disorder/H,0"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1

pbar=ProgressBar()
for n in pbar(range(100,297)):
    temp = np.load("./disorder_realizations/bond_disorder/H,"+str(n)+".npy")
    H_total = H_total + temp
    c=c+1
print(c)
H_total = H_total / c
print(H_total)
np.save("H_avg,bond_disorder,12",H_total)
# H_total = np.load("./H_avg,bond_disorder,12.npy")

e,u = np.linalg.eigh(H_total)

z=zm_state(2,1,pxp)
# z=ref_state(0,pxp)
z_energy = u[pxp.keys[z.ref],:]
t=np.arange(0,10,0.1)
f=np.zeros(np.size(t))
for n in range(0,np.size(f,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.show()

pbar=ProgressBar()
for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    z=ref_state(pxp.basis_refs[n],pxp)
    z_energy = u[pxp.keys[z.ref],:]
    f=np.zeros(np.size(t))
    for m in range(0,np.size(f,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[m])
        f[m] = np.abs(np.vdot(evolved_state,z_energy))**2
    plt.plot(t,f)
plt.title(r"$\textrm{Hypercube, Bond Disorder, Quench Fidelity}, N=12$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2 $")
plt.show()
plt.show()

z=zm_state(2,1,pxp)
z_energy = u[pxp.keys[z.ref],:]
overlap =np.log10(np.abs(z_energy)**2)
to_del=[]
eigvalues = np.copy(e)
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-5:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigvalues=np.delete(eigvalues,to_del[n])
    
plt.title(r"$\textrm{Hypercube, Bond Disorder, Eigenstate overlap with Neel}, N=12$")
plt.xlabel(r"$E$")
plt.ylabel(r"$\log \vert \langle E \vert \psi \rangle \vert^2$")
plt.scatter(eigvalues,overlap)
plt.show()

ent = entropy(pxp)
ent_vals = np.zeros(np.size(e))
for n in range(0,np.size(ent_vals,axis=0)):
    ent_vals[n] = ent.eval(u[:,n])
plt.title(r"$\textrm{Hypercube, Bond Disorder, Eigenstate Entropy}, N=12$")
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.scatter(e,ent_vals)
plt.show()
