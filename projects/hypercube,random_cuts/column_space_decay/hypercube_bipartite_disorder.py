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
N_main = 6
N_coupling = 6

pxp = unlocking_System([0,1],"periodic",2,N_main)
pxp_coupling = unlocking_System([0,1],"periodic",2,N_coupling)
pxp.gen_basis()
pxp_coupling.gen_basis()

H_main = spin_Hamiltonian(pxp,"x")
H_main.gen()

H_coupling = spin_Hamiltonian(pxp_coupling,"x")
H_coupling.gen()

plt.matshow(np.abs(H_main.sector.matrix()))
plt.show()

plt.matshow(np.abs(H_coupling.sector.matrix()))
plt.show()


H_total = np.zeros((2*pxp.dim+pxp_coupling.dim,2*pxp.dim+pxp_coupling.dim))

H_total[0:pxp.dim,0:pxp.dim] = H_main.sector.matrix()
H_total[pxp.dim:pxp.dim+pxp_coupling.dim,pxp.dim:pxp.dim+pxp_coupling.dim] = H_coupling.sector.matrix()
H_total[pxp.dim+pxp_coupling.dim:,pxp.dim+pxp_coupling.dim:] = H_main.sector.matrix()

#insert random connections into small hypercube tunnel
k=0.001
base_index = pxp.dim
for n in range(pxp.dim,pxp.dim+np.size(H_coupling.sector.matrix(),axis=0)):
    for m in range(0,pxp.dim):
        a=np.random.uniform(0,1)
        if a < k:
            H_total[n,m] = 1
            H_total[m,n] = 1
    for m in range(pxp.dim+pxp_coupling.dim,np.size(H_total,axis=0)):
        a=np.random.uniform(0,1)
        if a < k:
            H_total[n,m] = 1
            H_total[m,n] = 1

plt.matshow(np.abs(H_total))
plt.show()

e,u = np.linalg.eigh(H_total)
pbar=ProgressBar()

z=zm_state(2,1,pxp)
z_energy = np.conj(u[pxp.keys[z.ref],:])
t=np.arange(0,20,0.01)
f = np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.show()

t=np.arange(0,20,0.1)
for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    z=ref_state(pxp.basis_refs[n],pxp)
    z_energy = np.conj(u[pxp.keys[z.ref],:])
    f = np.zeros(np.size(t))
    for n in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[n])
        f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
    plt.plot(t,f)
plt.show()
