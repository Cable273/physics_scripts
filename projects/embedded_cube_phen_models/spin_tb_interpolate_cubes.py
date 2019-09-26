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

from Hamiltonian_Classes import Hamiltonian,H_table,clock_Hamiltonian,spin_Hamiltonian,H_operations
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

def spin_couplings(L):
    s=L/2
    m=np.arange(-s,s)
    couplings = np.power(s*(s+1)-m*(m+1),0.5)
    return couplings

L = 100
t=np.arange(0,20,0.01)

#custom chain with couplings from full/half etc
couplings_full = spin_couplings(L)
couplings_half = spin_couplings(L/2)
couplings_custom = np.zeros(np.size(couplings_half))
couplings_custom[0] = couplings_half[0]
N_max = int(L/2)-1
for n in range(1,int(L/2)):
    couplings_custom[n] = (-1/(N_max-1)*n+1/(N_max-1)+1)*couplings_half[n] + (1/(N_max-1)*n-1/(N_max-1))*couplings_full[n]
couplings_custom = np.append(couplings_custom,np.flip(couplings_custom))

H_custom = np.diag(couplings_custom,1)+np.diag(couplings_custom,-1)
e,u = np.linalg.eigh(H_custom)

# z_energy_fsa = np.conj(u_fsa[0,:])
z_energy = np.conj(u[0,:])
f=np.zeros(np.size(t))
# f_fsa=np.zeros(np.size(t))
pbar=ProgressBar()
for n in pbar(range(0,np.size(t,axis=0))):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    # evolved_state_prod = np.dot(u,evolved_state)
    # plt.plot(np.abs(evolved_state_prod)**2)
    # plt.axhline(y=1)
    # plt.tight_layout()
    # plt.xlabel(r"Hamming Distance")
    # plt.ylabel(r"$\vert \psi(n) \vert^2$")
    # plt.title(r"Interpolated Hypercubes, L="+str(L))

    # no_zeros = 5
    # if n != 0:
        # digits = int(np.log10(n))+1
    # else:
        # digits = 1
    # zeros_needed = no_zeros-digits
    # zeros=""
    # if zeros_needed >=1:
        # for m in range(0,zeros_needed):
            # zeros += "0"
    # label = zeros+str(n)
    # plt.savefig("./gif/img,"+label)
    # plt.cla()

    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2

for n in range(0,np.size(f,axis=0)):
    if f[n] < 0.1:
        cut = n
        break
max_index = np.argmax(f[cut:])
print(t[cut:][max_index],f[cut:][max_index])

overlap = np.log10(np.abs(u[0,:])**2)
plt.scatter(e,overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle 0 \vert E \rangle \vert^2)$")
plt.title(r"Interpolated Hypercubes, L="+str(L))
plt.show()
plt.plot(t,f,label="Interpolated Cube Chain")
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle \psi(0) \vert \psi(t) \rangle \vert^2$")
plt.title(r"Interpolated Hypercubes, L="+str(L))
plt.show()
