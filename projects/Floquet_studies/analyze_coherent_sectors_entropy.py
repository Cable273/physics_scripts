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
# pxp_syms = model_sym_data(pxp,[translational(pxp),parity(pxp),])
# pxp_syms = model_sym_data(pxp,[translational(pxp)])
# H = spin_Hamiltonian(pxp,"x",pxp_syms)
# k=[0]
# H.gen(k)
# H.sector.find_eig(k)

# U_mom = pxp_syms.basis_transformation(k)
# H_eigvalues = H.sector.eigvalues(k)
# H_eigvectors = H.sector.eigvectors(k)
# H = H.sector.matrix(k)

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
        else:
            thermalizing_states = np.vstack((thermalizing_states,states[:,n]))
            thermalizing_indices = np.append(thermalizing_indices,n)
thermalizing_states = np.delete(thermalizing_states,0,axis=0)
reviving_states = np.delete(reviving_states,0,axis=0)
oscillating_states = np.delete(oscillating_states,0,axis=0)
thermalizing_states = np.transpose(thermalizing_states)
reviving_states = np.transpose(reviving_states)
oscillating_states = np.transpose(oscillating_states)

oscillating_states_prod = np.dot(U_mom,oscillating_states)
reviving_states_prod = np.dot(U_mom,reviving_states)
thermalizing_states_prod = np.dot(U_mom,thermalizing_states)

oscillating_entropy = np.zeros(np.size(oscillating_states,axis=1))
reviving_entropy = np.zeros(np.size(reviving_states,axis=1))
thermalizing_entropy = np.zeros(np.size(thermalizing_states,axis=1))

oscillating_energy = np.zeros(np.size(oscillating_states,axis=1))
reviving_energy = np.zeros(np.size(reviving_states,axis=1))
thermalizing_energy = np.zeros(np.size(thermalizing_states,axis=1))

pbar=ProgressBar()
for n in pbar(range(0,np.size(oscillating_entropy,axis=0))):
    oscillating_entropy[n] = entropy(pxp).eval(oscillating_states_prod[:,n])
    oscillating_energy[n] = states_energy_exp[int(oscillating_indices[n])]
    # oscillating_energy[n] = np.vdot(oscillating_states[:,n],np.dot(H,oscillating_states[:,n]))
pbar=ProgressBar()
for n in pbar(range(0,np.size(reviving_entropy,axis=0))):
    reviving_entropy[n] = entropy(pxp).eval(reviving_states_prod[:,n])
    reviving_energy[n] = states_energy_exp[int(reviving_indices[n])]
    # reviving_energy[n] = np.vdot(reviving_states[:,n],np.dot(H,reviving_states[:,n]))
pbar=ProgressBar()
for n in pbar(range(0,np.size(thermalizing_entropy,axis=0))):
    thermalizing_entropy[n] = entropy(pxp).eval(thermalizing_states_prod[:,n])
    thermalizing_energy[n] = states_energy_exp[int(thermalizing_indices[n])]
    # thermalizing_energy[n] = np.vdot(thermalizing_states[:,n],np.dot(H,thermalizing_states[:,n]))

plt.scatter(oscillating_energy,oscillating_entropy,label="Oscillatory band")
plt.scatter(reviving_energy,reviving_entropy,label="Reviving band")
plt.scatter(thermalizing_energy,thermalizing_entropy,label="Thermalizing band")
plt.legend()
plt.xlabel(r"$E$")
plt.ylabel(r"$S$")
plt.title("Rydberg coherent state bipartite Entropy, N=24")
plt.show()

    

