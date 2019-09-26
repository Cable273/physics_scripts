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

L=50
d=3

#generate necessary spin chain H
bridge_cube_H = dict()
cube_dims = dict()
for n in range(2,L-d+2+1):
    bridge_cube_H[n] = spin_chain_tb_H(n)
    cube_dims[n] = np.size(bridge_cube_H[n],axis=0)
main_cube = spin_chain_tb_H(L)
cube_dims[L] = np.size(main_cube,axis=0)
keys = list(bridge_cube_H.keys())

#build block diagonal matrix, keep pair of indices for coupled sites + there coupling value
dim_total = 2*cube_dims[L]-1
for n in range(0,np.size(keys,axis=0)):
    dim_total = dim_total + (cube_dims[keys[n]] - 2) #-2 as sites shared by main cubes
H=np.zeros((dim_total,dim_total))

#two largest cubes coupled at pol
H_two_cubes = np.zeros((2*cube_dims[L]-1,2*cube_dims[L]-1))
H_two_cubes[:cube_dims[L],:cube_dims[L]] = main_cube
H_two_cubes[cube_dims[L]-1:,cube_dims[L]-1:] = main_cube

H[:np.size(H_two_cubes,axis=0),:np.size(H_two_cubes,axis=0)] = H_two_cubes

# get loc of start/end states of cube bridges
bridge_pos = dict()
for n in range(0,np.size(keys)):
    bridge_pos[keys[n]] = np.zeros(2,dtype=int)
    bridge_pos[keys[n]][0] = L+1 - keys[n]
    bridge_pos[keys[n]][1] =  2*L+1-bridge_pos[keys[n]][0]-1
    print(bridge_pos[keys[n]])

current_index = np.size(H_two_cubes,axis=0)
block_H = np.copy(bridge_cube_H[keys[0]])

#first bridge do sep
edge_coupling = block_H[0,1]
H[current_index,bridge_pos[keys[0]][0]] = edge_coupling
H[bridge_pos[keys[0]][0],current_index] = edge_coupling
H[current_index,bridge_pos[keys[0]][1]] = edge_coupling
H[bridge_pos[keys[0]][1],current_index] = edge_coupling
current_index = current_index + 1

plt.matshow(H)
plt.show()
# insert cube bridges, with last two couplings being at bridge pos
for n in range(0,np.size(keys,axis=0)):
    if keys[n] != 2: #do 1st coupling seperate
        block_H = np.copy(bridge_cube_H[keys[n]])
        edge_coupling = block_H[0,1]

        #trim first couplings (insert by hand with loc on main cube)
        block_H = np.delete(block_H,np.size(block_H,axis=0)-1,axis=0)
        block_H = np.delete(block_H,np.size(block_H,axis=1)-1,axis=1)
        block_H = np.delete(block_H,0,axis=0)
        block_H = np.delete(block_H,0,axis=1)

        dim = np.size(block_H,axis=0)
        H[current_index:current_index+dim,current_index:current_index+dim] = block_H

        H[current_index,bridge_pos[keys[n]][0]] = edge_coupling
        H[bridge_pos[keys[n]][0],current_index] = edge_coupling

        right_edge_index = current_index + dim -1

        H[right_edge_index,bridge_pos[keys[n]][1]] = edge_coupling
        H[bridge_pos[keys[n]][1],right_edge_index] = edge_coupling

        current_index = current_index + dim

plt.matshow(H)
plt.show()
print("Dim="+str(dim_total))
e,u = np.linalg.eigh(H)
print("Found eig")

t=np.arange(0,20,0.01)
psi_energy = np.conj(u[0,:])
eigenvalues = e
overlap = np.log10(np.abs(psi_energy)**2)
to_del=[]
for n in range(0,np.size(overlap,axis=0)):
    if overlap[n] <-15:
        to_del = np.append(to_del,n)
for n in range(np.size(to_del,axis=0)-1,-1,-1):
    overlap=np.delete(overlap,to_del[n])
    eigenvalues=np.delete(eigenvalues,to_del[n])
plt.scatter(eigenvalues,overlap)
plt.xlabel(r"$E$")
plt.ylabel(r"$\log(\vert \langle 0_L \vert E \rangle \vert^2)$")
plt.title(r"$\textrm{Cube Bridges Phenemological model}$, L="+str(L)+", d="+str(d))
plt.show()
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,psi_energy))**2
plt.plot(t,f)
plt.xlabel(r"$t$")
plt.ylabel(r"$\vert \langle 0_L \vert e^{-iHt} \vert 0_L \rangle \vert^2$")
plt.title(r"$\textrm{Cube Bridges Phenemological model}$, L="+str(L)+", d="+str(d))
plt.show()
