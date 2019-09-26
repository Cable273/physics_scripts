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
from Symmetry_Classes import translational,parity,model_sym_data,charge_conjugation,inversion
from Non_observables import zm
from Construction_functions import bin_to_int_base_m,int_to_bin_base_m,cycle_bits_state
from Search_functions import find_index_bisection
from State_Classes import zm_state,sym_state,prod_state,bin_state,ref_state
from rw_functions import save_obj,load_obj
from Calculations import level_stats,fidelity,eig_overlap,entropy,site_precession,site_projection,time_evolve_state
import itertools

def full_state_map(subsystem_bit_rep,site_labels,system,subsystem):
    full_prod_state_bits = np.zeros(system.N)
    for n in range(0,np.size(site_labels,axis=0)):
        full_prod_state_bits[site_labels[n]] = subsystem_bit_rep[n]
    full_ref = bin_to_int_base_m(full_prod_state_bits,system.base)
    return full_ref

#Form a state made from the polarized state |000...> with n singlets added.
#Form equal superposition over all possible ways of adding n singlets.
#Final state will be a |J,-J> state, with J=L/2 - n
def total_spin_state(J,system):
    s_system = 0.5*(system.base-1)
    J_max = system.N*s_system
    no_singlets = int(J_max-J)
    if no_singlets == 0:
        psi = np.zeros(np.size(system.basis_refs))
        psi[0] = 1
        return psi
    else:
        #form simplest subsystem wf of n singlets next to each other
        subsystem = unlocking_System([0,1],"periodic",2,2*no_singlets)
        if no_singlets>1:
            psi_sub0 = np.zeros(np.size(subsystem.basis_refs))
            for n in range(0,np.size(subsystem.basis_refs,axis=0)):
                state = np.copy(subsystem.basis[n])
                phase=1
                for m in range(0,np.size(state,axis=0),2):
                    temp = np.array((state[m],state[m+1]))
                    if (temp == np.array((0,0))).all() or (temp == np.array((1,1))).all():
                        phase = 0
                        break
                    elif (temp == np.array((0,1))).all():
                        phase = phase * -1
                psi_sub0[n] = phase
        else:
            psi_sub0 = np.zeros(np.size(subsystem.basis_refs))
            psi_sub0[2] = 1
            psi_sub0[1] = -1
        psi_sub0 = psi_sub0/np.power(np.vdot(psi_sub0,psi_sub0),0.5)

        #get all locations (relabelling) singlets can be located
        sites = np.arange(0,system.N)
        singlet_sites = np.array((list(itertools.combinations(sites,2*no_singlets))))
        from combinatorics import all_pairs
        poss_singlet_loc=np.zeros((np.size(singlet_sites,axis=0),no_singlets*2),dtype=int)

        for n in range(0,np.size(singlet_sites,axis=0)):
            for pairs in all_pairs(list(singlet_sites[n])):
                set_of_pairs = np.ndarray.flatten(np.array((pairs)))
                poss_singlet_loc[n] = set_of_pairs


        # #form binary prod states from psi_sub0 and permuting.
        # #|J,-J> formed by equal superposition of all states, remaining sites 0
        psi = np.zeros(np.size(system.basis_refs))
        print("Forming |J,-J> state, J=",J)
        pbar=ProgressBar()
        for n in pbar(range(0,np.size(poss_singlet_loc,axis=0))):
            for m in range(0,np.size(psi_sub0,axis=0)):
                ref = full_state_map(subsystem.basis[m],poss_singlet_loc[n],system,subsystem)
                psi[system.keys[ref]] = psi[system.keys[ref]] + psi_sub0[m]

        psi = psi/np.power(np.vdot(psi,psi),0.5)
        return psi

system = unlocking_System([0,1],"periodic",2,6)
J=0
psi = total_spin_state(J,system)

