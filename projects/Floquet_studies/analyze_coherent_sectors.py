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

pxp = unlocking_System("pxp",[0],"periodic",2,24)
# # pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
# H = spin_Hamiltonian(pxp,"x",pxp_syms)
# k=[0]
# H.gen(k)
# H.sector.find_eig(k)
# np.save("pxp,H,k0,24.npy",H.sector.matrix(k))
# np.save("pxp,eigvalues,k0,24.npy",e)
# np.save("pxp,eigvectors,k0,24.npy",u)

print("Loading data")
H = np.load("./pxp,H,k0,24.npy")
H_eigvalues = np.load("./pxp,eigvalues,k0,24.npy")
H_eigvectors = np.load("./pxp,eigvectors,k0,24.npy")
U_mom = np.load("./U_mom,k0,pxp,24.npy")
print("Loaded")

t_range = np.arange(0,20,0.01)
states = np.load("./data/24/cstates,k0,24.npy")
states_energy_exp = np.zeros(np.size(states,axis=1))

therm_energies = np.array((-8.29610026,8.95347315,7.28241295))
def is_energy_allowed(energy,forbidden_energy):
    allowed = 1
    for n in range(0,np.size(forbidden_energy,axis=0)):
        if np.abs(energy - forbidden_energy[n])<1e-5:
            allowed = 0
            break
    if allowed == 1:
        return True
    else:
        return False

t = np.arange(0,20,0.01)
pbar=ProgressBar()
thermalizing_states = np.zeros(np.size(states,axis=0))
reviving_states = np.zeros(np.size(states,axis=0))
oscillating_states = np.zeros(np.size(states,axis=0))
thermalizing_indices = []
reviving_indices = []
oscillating_indices = []
for n in pbar(range(0,np.size(states,axis=1))):
    states_energy_exp[n] = np.vdot(states[:,n],np.dot(H,states[:,n]))
    temp = np.load("./data/24/fidelity/cstate_fidelity,"+str(n)+".npy")
    #never decay to zero => oscillating_states
    if (temp<0.001).any() == False:
        oscillating_states = np.vstack((oscillating_states,states[:,n]))
        oscillating_indices = np.append(oscillating_indices,n)
    else:
        #calc integral of F. Non reviving states will have small value
        temp_integral = np.trapz(temp)
        if temp_integral>60 and is_energy_allowed(states_energy_exp[n],therm_energies):
            reviving_states = np.vstack((reviving_states,states[:,n]))
            reviving_indices = np.append(reviving_indices,n)
            plt.plot(t,temp)
        else:
            thermalizing_states = np.vstack((thermalizing_states,states[:,n]))
            thermalizing_indices = np.append(thermalizing_indices,n)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle \psi(t) \vert \psi(0) \rangle \vert^2$")
# plt.title("Thermalizing Rydberg coherent state Fidelity, PXP, N=24")
# plt.title("Oscillating Rydberg coherent state Fidelity, PXP, N=24")
plt.title("Reviving Rydberg coherent state Fidelity, PXP, N=24")
plt.show()
thermalizing_states = np.delete(thermalizing_states,0,axis=0)
reviving_states = np.delete(reviving_states,0,axis=0)
oscillating_states = np.delete(oscillating_states,0,axis=0)
thermalizing_states = np.transpose(thermalizing_states)
reviving_states = np.transpose(reviving_states)
oscillating_states = np.transpose(oscillating_states)

#oscillating state overlap with eigenstates
oscillating_energies=np.zeros(np.size(oscillating_states,axis=1))
oscillating_overlaps=np.zeros(np.size(oscillating_states,axis=1))
for n in range(0,np.size(oscillating_states,axis=1)):
    max_overlap = 0
    for m in range(0,np.size(H_eigvectors,axis=1)):
        temp = np.abs(np.vdot(H_eigvectors[:,m],oscillating_states[:,n]))**2
        if temp > max_overlap:
            max_overlap = temp
    oscillating_overlaps[n] = max_overlap
    oscillating_energies[n] = states_energy_exp[int(oscillating_indices[n])]
plt.scatter(oscillating_energies,oscillating_overlaps)
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \alpha \vert E \rangle \vert^2$")
plt.title("Oscillating Rydberg coherent state, maximum overlap with PXP eigenstates, N="+str(pxp.N))
plt.show()

#reviving state overlap with Neel state and anti_neel
z=zm_state(2,1,pxp)
reviving_states_prod = np.dot(U_mom,reviving_states)
reviving_overlaps=np.zeros(np.size(reviving_states,axis=1))
reviving_energies=np.zeros(np.size(reviving_states,axis=1))
for n in range(0,np.size(reviving_states,axis=1)):
    reviving_overlaps[n] = np.abs(reviving_states_prod[pxp.keys[z.ref],n])**2
    reviving_energies[n] = states_energy_exp[int(reviving_indices[n])]
print(np.sort(reviving_energies))
plt.scatter(reviving_energies,reviving_overlaps,label="Neel")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \alpha \vert Z_2 \rangle \vert^2$")
plt.title("Reviving Rydberg coherent state, overlap with Neel state, N="+str(pxp.N))
plt.legend()
plt.show()

#thermalizing states largest product state overlap
thermalizing_states_prod = np.dot(U_mom,thermalizing_states)
thermalizing_overlaps=np.zeros(np.size(thermalizing_states,axis=1))
thermalizing_energies=np.zeros(np.size(thermalizing_states,axis=1))
prods_indices = []
for n in range(0,np.size(thermalizing_states,axis=1)):
    max_index = np.argmax(thermalizing_states_prod[:,n])
    prods_indices = np.append(prods_indices,n)
    # thermalizing_overlaps[n] = np.abs(thermalizing_states_prod[max_index,n])**2
    thermalizing_overlaps[n] = np.abs(thermalizing_states_prod[0,n])**2
    print(thermalizing_overlaps[n],pxp.basis[max_index])
    thermalizing_energies[n] = states_energy_exp[int(thermalizing_indices[n])]
plt.scatter(thermalizing_energies,thermalizing_overlaps,label="Neel")
plt.xlabel(r"$E$")
plt.ylabel(r"$\vert \langle \alpha \vert Z_2 \rangle \vert^2$")
plt.title(r"Thermalizing Rydberg coherent state, overlap with $\vert 000... \rangle$, N="+str(pxp.N))
plt.legend()
plt.show()
