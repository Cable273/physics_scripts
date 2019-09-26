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

def spin_chain_tb_H(L):
    S=L/2
    m=np.arange(-S,S)
    couplings = np.power(S*(S+1)-m*(m+1),0.5)
    return np.diag(couplings,1) + np.diag(couplings,-1)

L=15
d=3

#generate necessary spin chain H
bridge_cube_H = spin_chain_tb_H(L-2)
main_cube = spin_chain_tb_H(L)
bridge_dim = int(np.size(bridge_cube_H,axis=0))
main_dim = int(np.size(main_cube,axis=0))

# #build block diagonal matrix, keep pair of indices for coupled sites + there coupling value
dim_total = 2*main_dim-1
dim_total = dim_total + (L+1-3)*(bridge_dim-2)
H=np.zeros((dim_total,dim_total))

# #two largest cubes coupled at pol
H_two_cubes = np.zeros((2*main_dim-1,2*main_dim-1))
H_two_cubes[:main_dim,:main_dim] = main_cube
H_two_cubes[main_dim-1:,main_dim-1:] = main_cube

H[:np.size(H_two_cubes,axis=0),:np.size(H_two_cubes,axis=0)] = H_two_cubes

# get loc of start/end states of cube bridges
bridge_pos = np.zeros((L+1-3,2),dtype=int)
for n in range(2,L):
    bridge_pos[n-2,0] = n
    bridge_pos[n-2,1] = n+L-1
print(bridge_pos)

current_index = np.size(H_two_cubes,axis=0)
block_H = np.copy(bridge_cube_H)

plt.matshow(H)
plt.show()

# insert cube bridges, with last two couplings being at bridge pos
block_H = np.copy(bridge_cube_H)
edge_coupling = bridge_cube_H[0,1]
#trim first couplings (insert by hand with loc on main cube)
block_H = np.delete(block_H,np.size(block_H,axis=0)-1,axis=0)
block_H = np.delete(block_H,np.size(block_H,axis=1)-1,axis=1)
block_H = np.delete(block_H,0,axis=0)
block_H = np.delete(block_H,0,axis=1)
dim = np.size(block_H,axis=0)

for n in range(0,np.size(bridge_pos,axis=0)):
    H[current_index:current_index+dim,current_index:current_index+dim] = block_H

    H[current_index,bridge_pos[n,0]] = edge_coupling
    H[bridge_pos[n,0],current_index] = edge_coupling

    right_edge_index = current_index + dim -1

    H[right_edge_index,bridge_pos[n,1]] = edge_coupling
    H[bridge_pos[n,1],right_edge_index] = edge_coupling

    current_index = current_index + dim

plt.matshow(H)
plt.show()
print("Dim="+str(dim_total))
e,u = np.linalg.eigh(H)
print("Found eig")

t=np.arange(0,20,0.01)
psi_energy = np.conj(u[0,:])
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle 0_L \vert e^{-iHt} \vert 0_L \rangle \vert^2$")
plt.title(r"$\textrm{Cube Bridges Phenemological model, Unequal bridge length}$, L="+str(L)+", d="+str(d))
plt.show()
