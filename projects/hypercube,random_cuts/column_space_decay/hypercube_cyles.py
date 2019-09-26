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
pxp = unlocking_System([0,1],"periodic",2,6)
pxp.gen_basis()

#form hamming rep starting from Neel
z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)

# #remove half of hypercube graph
# to_remove_refs = []
# for n in range(int(pxp.N/2)+1,len(hamming_sectors)):
    # to_remove_refs = np.append(to_remove_refs,hamming_sectors[n])

# for n in range(0,np.size(to_remove_refs,axis=0)):
    # if np.abs(to_remove_refs[n] )<1e-10:
        # print("ZERO DELETED")
# #redo basis
# pxp.basis_refs_new = np.zeros(np.size(pxp.basis_refs)-np.size(to_remove_refs))
# c=0
# for n in range(0,np.size(pxp.basis_refs,axis=0)):
    # if pxp.basis_refs[n] not in to_remove_refs:
        # pxp.basis_refs_new[c] = pxp.basis_refs[n]
        # c = c+1
# pxp.basis_refs = pxp.basis_refs_new

# pxp.basis = np.zeros((np.size(pxp.basis_refs),pxp.N))
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.basis[n] = int_to_bin_base_m(pxp.basis_refs[n],pxp.base,pxp.N)
# pxp.keys = dict()
# for n in range(0,np.size(pxp.basis_refs)):
    # pxp.keys[int(pxp.basis_refs[n])] = n
# pxp.dim = np.size(pxp.basis_refs)

z=zm_state(2,1,pxp)
hamming_sectors = find_hamming_sectors(z.bits)
hamming_length = 0
for n in range(0,len(hamming_sectors)):
    if np.size(hamming_sectors[n])!=0:
        hamming_length = hamming_length + 1
        print("\n")
        for m in range(0,np.size(hamming_sectors[n],axis=0)):
            print(pxp.basis[pxp.keys[hamming_sectors[n][m]]])
            

H=spin_Hamiltonian(pxp,"x")
H.gen()

#insert N_c cycles with k_int random coupling rate to other cycles and k_cube random coupling rate to hypercube, ending on polarized
N_c = 10
# k_int = 1e-2
k_cube = 0.3

neel_index = pxp.keys[z.ref]
pol_index = 0

H_cube_dim = pxp.dim

cycle_hopping_H = np.diag(np.ones(hamming_length-1),1)+np.diag(np.ones(hamming_length-1),-1)
cycle_H = np.zeros((N_c*hamming_length,N_c*hamming_length))
init_coord=0
for n in range(0,N_c):
    cycle_H[init_coord:(init_coord+hamming_length),init_coord:(init_coord+hamming_length)] = cycle_hopping_H
    init_coord  = init_coord + hamming_length
 
e,u = np.linalg.eigh(cycle_H)
psi_energy = np.conj(u[0,:])
t=np.arange(0,20,0.01)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(psi_energy,e,t[n])
    f[n]=np.abs(np.vdot(evolved_state,psi_energy))**2

dim_total = H_cube_dim + np.size(cycle_H,axis=0)
H_total = np.zeros((dim_total,dim_total))
H_total[0:H_cube_dim,0:H_cube_dim] = H.sector.matrix()
H_total[H_cube_dim:,H_cube_dim:] = cycle_H
print("total dim="+str(dim_total))

index = H_cube_dim
for n in range(0,N_c):
    end_ref = np.random.choice(hamming_sectors[hamming_length-1])
    end_ref = 0
    print(end_ref)
    end_index = pxp.keys[end_ref]
    H_total[neel_index,index+1] = 1
    H_total[end_index,index+hamming_length-1] = 1

    H_total[index+1,neel_index] = 1
    H_total[index+hamming_length-1,end_index] = 1
    index = index + hamming_length
    
# plt.matshow(np.abs(H_total))
# plt.show()

# insert couplings randomly with rates k_int,k_cube
cycle_first_indices = np.zeros(N_c)
cycle_first_indices[0] = H_cube_dim
for n in range(1,N_c):
    cycle_first_indices[n]  =cycle_first_indices[n-1] + hamming_length

# for n in range(0,np.size(cycle_first_indices)):
    # for m in range(1,hamming_length-1):
        # a = np.random.uniform(0,1)
        # if a<=k_cube:
            # availiable_states = np.append(hamming_sectors[m-1],hamming_sectors[m+1])
            # ref = np.random.choice(availiable_states)
            # ref_index = pxp.keys[ref]
            # H_total[ref_index,int(cycle_first_indices[n]+m)] = 1
            # H_total[int(cycle_first_indices[n]+m),ref_index] = 1

# for n in range(0,np.size(H_total,axis=0)):
    # for m in range(0,H_cube_dim):
        # a = np.random.uniform(0,1)
        # if a <=k_int:
            # H_total[n,m] = 1
            # H_total[m,n] = 1

# for n in range(H_cube_dim+1,np.size(H_total,axis=0)):
    # for m in range(H_cube_dim+1,np.size(H_total,axis=0)):
        # a = np.random.uniform(0,1)
        # if a <=k_cube:
            # H_total[n,m] = 1
            # H_total[m,n] = 1

plt.matshow(np.abs(H_total))
plt.show()

e,u = np.linalg.eigh(H_total)

z=zm_state(2,1,pxp)
# z=ref_state(1,pxp)
z_index = pxp.keys[z.ref]
z_energy = np.conj(u[z_index,:])
t=np.arange(0,10,0.1)
f=np.zeros(np.size(t))
for n in range(0,np.size(t,axis=0)):
    evolved_state = time_evolve_state(z_energy,e,t[n])
    f[n] = np.abs(np.vdot(evolved_state,z_energy))**2
plt.plot(t,f)
plt.show()

pbar=ProgressBar()
for n in pbar(range(0,np.size(pxp.basis_refs,axis=0))):
    z=ref_state(pxp.basis_refs[n],pxp)
    z_index = pxp.keys[z.ref]
    z_energy = np.conj(u[z_index,:])
    t=np.arange(0,10,0.1)
    f=np.zeros(np.size(t))
    for m in range(0,np.size(t,axis=0)):
        evolved_state = time_evolve_state(z_energy,e,t[m])
        f[m] = np.abs(np.vdot(evolved_state,z_energy))**2
    plt.plot(t,f)
plt.show()
    # for m in range(0,np.size(f,axis=0)):
        # if f[m]<0.1:
            # cut = n
            # break
    # f_max = np.max(f[cut:])
    # if f_max > 0.6:
        # print(pxp.basis[n])
        # overlap = np.log10(np.abs(z_energy)**2)
        # plt.scatter(e,overlap)
        # plt.show()

